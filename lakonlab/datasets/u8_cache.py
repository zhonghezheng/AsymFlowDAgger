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
import os.path as osp

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
