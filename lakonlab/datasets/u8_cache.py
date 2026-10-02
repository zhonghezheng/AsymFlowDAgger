"""Read-only access to a preprocessed uint8 image cache (see
tools/build_imagenet_u8_cache.py).

The cache holds every training image AFTER ``image_preproc(img, size,
random_flip=False)`` -- the exact array the live JPEG loaders produce -- as one
``[N, H, W, 3]`` uint8 memmap, sorted by class. Loading a bank then costs a slice
copy instead of a JPEG decode + resize (PIL, GIL-bound: the measured bottleneck of
bank construction), and a whole-class bank is one contiguous read.

uint8 is exact, not an approximation: the diffusion input is an affine function of
uint8/255 (``patchify(vae.encode(img * 2 - 1))`` with the identity RGBColorEncoder),
so nothing is lost by not storing floats.
"""
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class U8ImageCache:
    """``prefix`` names the three files the builder writes:
    ``<prefix>.u8.npy`` (the memmap), ``<prefix>.index.npz`` (relpaths / labels /
    class offsets) and ``<prefix>.complete`` (written last, only on success).

    Opened lazily on first use, so an instance built in the parent survives fork and
    each process maps the file itself. Reads are thread-safe (read-only memmap).
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self._mm = None
        self._row = None
        self._fd = None
        self._pool = None
        self._pool_key = None   # (pid, threads) the pool was made for

    def _open(self):
        if self._mm is not None:
            return
        done = self.prefix + '.complete'
        if not osp.exists(done):
            # a partially built cache has unwritten rows that read back as zeros --
            # black images that would enter the banks silently. Refuse instead.
            raise FileNotFoundError(
                f'u8 cache {self.prefix!r} is not complete (no {done}). Finish '
                f'tools/build_imagenet_u8_cache.py or unset the cache path.')
        idx = np.load(self.prefix + '.index.npz', allow_pickle=False)
        self._row = {p: i for i, p in enumerate(idx['relpaths'].tolist())}
        self._mm = np.load(self.prefix + '.u8.npy', mmap_mode='r')
        # a plain fd beside the memmap for read_into's pread path (offset-explicit, so
        # one fd is safe across threads and across fork)
        self._fd = os.open(self.prefix + '.u8.npy', os.O_RDONLY)
        # once per process, so a run's log says which loader its banks/targets used
        from lakonlab.utils import get_root_logger
        get_root_logger().info(
            f'u8 image cache in use: {self.prefix} ({self._mm.shape[0]:,} images, '
            f'{self._mm.shape[1]}x{self._mm.shape[2]}; flips applied at read time)')

    def get(self, rel_paths, flip=None):
        """uint8 ``[M, H, W, 3]`` CPU tensor for ``rel_paths``, in the GIVEN order.

        ``flip`` (bool array ``[M]``, optional) mirrors those rows horizontally --
        the same ``arr[:, ::-1]`` image_preproc applies under random_flip.
        """
        self._open()
        rows = np.fromiter((self._row[p] for p in rel_paths), dtype=np.int64,
                           count=len(rel_paths))
        # read in ascending row order (a whole-class bank becomes one sequential
        # run), then restore the caller's order
        order = np.argsort(rows, kind='stable')
        out = np.empty((len(rows), ) + self._mm.shape[1:], dtype=np.uint8)
        out[order] = self._mm[rows[order]]
        if flip is not None:
            flip = np.asarray(flip, dtype=bool)
            if flip.any():
                out[flip] = out[flip][:, :, ::-1]
        return torch.from_numpy(out)

    @property
    def image_shape(self):
        """``(H, W, 3)`` of one cached image."""
        self._open()
        return tuple(self._mm.shape[1:])

    def read_into(self, rel_paths, out, threads=8):
        """Fill ``out`` -- a C-contiguous uint8 array ``[len(rel_paths), H, W, 3]``,
        e.g. a reused (pinned) bank buffer -- with ``rel_paths``' images, in the GIVEN
        order. Bit-identical to :meth:`get` (same file bytes, same order).

        One ``pread`` per image straight into ``out`` on ``threads`` threads: the copy
        runs in the kernel with the GIL released, so the threads are truly parallel,
        and nothing is written twice (``get``'s gather + reorder copies every byte
        twice on one thread). Measured with all 8 ranks of a node loading at once:
        ~1.7 -> ~9-11 GB/s per rank at 8 threads into a reused buffer; more threads
        stop helping there -- the node's memory bandwidth is the ceiling.
        """
        self._open()
        n = len(rel_paths)
        assert (out.dtype == np.uint8 and out.flags.c_contiguous
                and tuple(out.shape) == (n, ) + tuple(self._mm.shape[1:])), (
            out.dtype, out.shape, self._mm.shape)
        rows = [self._row[p] for p in rel_paths]
        rb = out[0].nbytes if n else 0
        off0, fd = self._mm.offset, self._fd

        def run(lo, hi):
            for k in range(lo, hi):
                got = os.preadv(fd, [memoryview(out[k]).cast('B')], off0 + rows[k] * rb)
                if got != rb:
                    raise IOError(f'short read from {self.prefix}.u8.npy: row {rows[k]} '
                                  f'({got} of {rb} bytes)')

        threads = max(1, min(int(threads), n))
        if threads == 1:
            run(0, n)
            return out
        key = (os.getpid(), threads)       # pools do not survive fork
        if self._pool_key != key:
            self._pool = ThreadPoolExecutor(max_workers=threads)
            self._pool_key = key
        bounds = np.linspace(0, n, threads + 1).astype(int)
        # list() re-raises any worker's exception here
        list(self._pool.map(run, bounds[:-1], bounds[1:]))
        return out
