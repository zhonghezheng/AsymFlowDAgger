# Copyright (c) 2026 Hansheng Chen

from io import BytesIO
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmcv.fileio import FileClient

from ...builder import MODULES
from lakonlab.datasets.imagenet import image_preproc


def _load_one_u8_worker(args):
    """Module-level (and NOT registered) so a process pool could pickle it: a bound
    method of an expert holding tensors and a FileClient is not picklable. Returns
    uint8 HWC. Kept module-level now that the loader is threaded, since the split
    between path picking and decoding is still the useful shape."""
    import os.path as _osp
    data_root, rel_path, image_size = args
    img = Image.open(_osp.join(data_root, rel_path)).convert('RGB')
    return torch.from_numpy(image_preproc(img, image_size, random_flip=False))


class U8Bank:
    """A bank held as its uint8 ORIGINALS plus the scoring features of every entry
    (``bank_storage='u8'``).

    ``u8`` is the loader's CPU tensor ``[N, H, W, 3]``, kept as-is; ``feats`` is
    ``[n_orient * N, Df]`` on the device, laid out ``[originals; flips]`` exactly as
    the bf16-latent bank's rows are. The latents themselves are NOT stored:
    :meth:`chunks` re-derives them on the GPU with the same ops
    :meth:`EmpiricalExpert._finish_bank` runs -- ``/255``, the h-flip in IMAGE
    space, then ``encode_fn`` -- so every entry, flips included, is the exact fp32
    latent rather than a bf16 rounding of it, nothing ever crosses back to the host,
    and a streamed chunk is 1 byte per element for both orientations instead of 2
    bytes per element per orientation.
    """
    __slots__ = ('u8', 'feats', 'encode_fn', 'n_orient', 'lease')

    def __init__(self, u8, feats, encode_fn, n_orient, lease=None):
        assert feats.shape[0] == n_orient * u8.shape[0], (feats.shape, u8.shape, n_orient)
        self.u8 = u8
        self.feats = feats
        self.encode_fn = encode_fn
        self.n_orient = n_orient
        # (HostBankBuffer, generation) when ``u8`` is a view into a REUSED buffer:
        # checked on every read, so a bank whose buffer was released (and possibly
        # refilled with another band's images) raises instead of reading them.
        self.lease = lease

    def chunks(self, device, chunk):
        """Yield ``(lo, hi, latents)``: bank entries ``lo:hi`` as fp32 latents on
        ``device``. Each uint8 chunk crosses to the GPU ONCE and serves every
        orientation; ``chunk`` counts entries per transfer, across orientations."""
        if self.lease is not None:
            self.lease[0].check(self.lease[1])
        n = self.u8.shape[0]
        step = max(1, chunk // self.n_orient)
        for b0 in range(0, n, step):
            b1 = min(b0 + step, n)
            imgs = self.u8[b0:b1].to(device, non_blocking=True).permute(
                0, 3, 1, 2).float() / 255.0
            for o in range(self.n_orient):
                x = torch.flip(imgs, dims=[-1]) if o else imgs
                yield o * n + b0, o * n + b1, self.encode_fn(x).float()

    def entry(self, k, device):
        """Bank entry ``k`` (index into the ``[originals; flips]`` layout) as an fp32
        latent ``[C, H, W]``, derived with exactly the ops :meth:`chunks` uses."""
        if self.lease is not None:
            self.lease[0].check(self.lease[1])
        n = self.u8.shape[0]
        o, j = divmod(int(k), n)
        img = self.u8[j:j + 1].to(device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.flip(img, dims=[-1]) if o else img
        return self.encode_fn(x).float()[0]

    def orientations(self, k):
        """Every entry index holding the same IMAGE as entry ``k`` (it and its flip)."""
        n = self.u8.shape[0]
        j = int(k) % n
        return [o * n + j for o in range(self.n_orient)]


class HostBankBuffer:
    """One reusable host buffer holding a whole band's banks (uint8 images), pinned
    when possible so both uploads -- :meth:`EmpiricalExpert._finish_bank` and the
    x0_hat stream -- run at pinned speed (measured 27 vs 9.7 GB/s for pageable).

    Reuse never lets a band see another band's images:
      * each bank is a view of EXACTLY the rows loaded for it (consecutive,
        non-overlapping slices), so rows left over from an earlier, larger band are
        never part of any bank;
      * a buffer is handed out only while not ``busy``: it is leased when a band's
        draw is issued and released only once that band is over -- after a device
        sync, so no asynchronous upload is still reading it -- or, for a draw that
        is abandoned, once its load has actually finished writing;
      * every lease bumps ``gen``, and every read of a bank checks its lease is still
        the live one (:meth:`check`), so a stale bank raises rather than silently
        reading the images of whichever band holds the buffer now.

    Allocated and (un)registered on the main thread only -- the prefetch thread just
    writes into it with ``pread``, so no CUDA call is ever made off-thread.
    """
    HEADROOM = 1.25    # grow to 1.25x what was asked, so band-to-band size jitter
                       # (the scored-trajectory count varies) does not re-register

    def __init__(self, image_shape, pin):
        self.image_shape = tuple(image_shape)
        self.pin = pin
        self.arr = None          # numpy uint8 [cap, H, W, 3]
        self.tensor = None       # torch view of arr
        self.cap = 0
        self.pinned = False
        self.gen = 0
        self.busy = False

    def lease(self, n):
        """Lease the buffer for ``n`` images. Returns the lease's generation."""
        assert not self.busy, 'HostBankBuffer leased while still in use'
        if n > self.cap:
            self._reallocate(int(n * self.HEADROOM) + 1)
        self.gen += 1
        self.busy = True
        return self.gen

    def release(self, gen):
        if self.gen == gen:
            self.busy = False

    def check(self, gen):
        assert self.busy and self.gen == gen, (
            'a bank was read after its HostBankBuffer was released or re-leased -- '
            'its images may already belong to another band')

    def _reallocate(self, cap):
        self.free()
        arr = np.empty((cap, ) + self.image_shape, np.uint8)
        if self.pin and torch.cuda.is_available():
            # registration faults every page in (pinning needs them resident), so it
            # doubles as the pre-touch that spares the loads their first-touch faults
            err = torch.cuda.cudart().cudaHostRegister(arr.ctypes.data, arr.nbytes, 0)
            self.pinned = int(err) == 0
        if not self.pinned:
            arr.fill(0)          # pre-touch: a fresh buffer pays a fault per page
        self.arr, self.tensor, self.cap = arr, torch.from_numpy(arr), cap

    def free(self):
        if self.arr is not None and self.pinned:
            torch.cuda.synchronize()     # no upload may still be reading it
            torch.cuda.cudart().cudaHostUnregister(self.arr.ctypes.data)
        self.arr = self.tensor = None
        self.cap = 0
        self.pinned = False

    @property
    def nbytes(self):
        return 0 if self.arr is None else self.arr.nbytes


@MODULES.register_module()
class EmpiricalExpert(nn.Module):
    """Per-class posterior-mean data oracle backed by the **dataset on disk**.

    There is never a ground-truth ``x_0`` for a visited rollout state -- the expert
    returns the Bayes posterior *mean* over a per-class bank of real images under
    the codebase convention

        x_t = (1 - sigma) * x_0 + sigma * noise      (x_0 = data, sigma in [0, 1])

    so for any visited state ``x_t`` at level ``sigma``:

        w_i  propto  exp(-|| feat(x_t) - (1 - sigma) * feat(x_0^i) ||^2 / (2 sigma^2))
        x0_hat = sum_i w_i * x_0^i

    Unlike a fixed RAM reservoir, the bank is drawn fresh from the FULL dataset:
    the ``datalist`` gives ``class -> [image paths]``, and :meth:`build_banks`
    loads ``bank_size`` random images per requested class, encodes them with the
    caller's ``encode_fn`` (the exact train pipeline: ``patchify(vae.encode(img*2-1))``)
    and scores them in the compact ``feat_fn`` subspace. This removes the ~TB RAM
    reservoir and lets the bank cover the whole class (not a sliding window), at the
    cost of amortized disk IO per rollout round. Banks are built per rollout chunk
    and discarded, so peak memory is ``chunk_classes * bank_size`` latents.

    The null / unconditional class (id ``>= num_classes``) draws from the whole
    dataset. Everything is per-process (each DDP rank loads its own banks) and never
    checkpointed (no parameters / buffers).

    Args:
        datalist_path (str): whitespace ``<rel_path> <label>`` list (train.txt).
        data_root (str): root the ``rel_path`` entries are joined to.
        image_size (int): center-crop size (must match the train pipeline).
        num_classes (int): number of real classes.
        bank_size (int | None): images drawn per class per bank (the posterior
            support). ``None`` uses the ENTIRE class (all its images).
        null_bank_size (int): cap for the null / unconditional bank (drawn from the
            whole dataset) -- always capped, since the unconditional posterior is
            just the global mean and loading 1.28M images would be infeasible.
        null_bank_mode (str): ``'per_traj'`` (default) draws an independent
            ``null_bank_size`` bank for EVERY trajectory; ``'shared'`` draws ONE
            ``null_bank_size`` bank per bank build, shared by all of its trajectories
            -- e.g. 20k images seen by every row instead of 2k unique per row, for
            the IO of ~10 per-trajectory banks.
        num_workers (int): threads for the on-disk image load/decode (GIL-released).
        sample_chunk (int): rows processed at once in x0_hat (bounds memory).
        bank_chunk (int): bank entries summed at once (bounds peak memory).
        min_sigma (float): floor on ``sigma`` in the posterior denominator.
        bank_storage (str): how a finished bank holds its entries. ``'u8'``
            (default) keeps the uint8 originals on the host and re-derives each
            latent chunk on the GPU inside :meth:`x0_hat` (see :class:`U8Bank`):
            exact, 4x fewer bytes streamed, no device->host copy at build. That
            re-runs ``encode_fn`` per streamed chunk, which is free for the identity
            RGBColorEncoder (an affine + reshape) but would re-run a real VAE on
            every call -- use ``'latent'`` there: bf16 latents (both orientations)
            encoded once at build and copied to the host.
        u8_read_threads (int): ``pread`` threads per process for u8-cache bank loads
            (see :meth:`U8ImageCache.read_into`). 8 measured best with 8 ranks per
            node loading at once; 1 is the old single-threaded rate.
        host_buffers (int): reusable host buffers for PREFETCHED banks (see
            :class:`HostBankBuffer`); the on-policy band needs one, a second covers
            an abandoned draw whose load is still running. 0 -> a fresh allocation
            per band, as before. Needs ``u8_cache``.
        pin_host_buffers (bool): page-lock those buffers for pinned-speed uploads.
    """

    def __init__(self,
                 datalist_path,
                 data_root,
                 image_size=256,
                 num_classes=1000,
                 bank_size=256,
                 null_bank_size=512,
                 null_bank_mode='per_traj',
                 null_temp=1.0,
                 temp_spread=None,
                 temp_spread_null_only=False,
                 temp_spread_null=None,
                 random_flip=True,
                 include_flips=False,
                 num_workers=32,
                 sample_chunk=32,
                 bank_chunk=128,
                 min_sigma=1e-3,
                 kernel_space='feat',
                 u8_cache=None,
                 bank_storage='u8',
                 u8_read_threads=8,
                 host_buffers=2,
                 pin_host_buffers=True):
        super().__init__()
        assert bank_storage in ('u8', 'latent'), bank_storage
        self.bank_storage = bank_storage
        self.u8_read_threads = int(u8_read_threads)
        self.host_buffers = int(host_buffers)
        self.pin_host_buffers = bool(pin_host_buffers)
        self._host_bufs = []      # HostBankBuffer pool, created on first lease
        self.datalist_path = datalist_path
        self.data_root = data_root
        self.image_size = image_size
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.null_bank_size = null_bank_size
        # 'shared': one null draw per build, referenced by every trajectory. Exact for
        # x0_hat (which only reads a bank), and x0_hat streams a bank once for all the
        # rows that share it, so a large shared bank costs one pass, not one per row.
        assert null_bank_mode in ('per_traj', 'shared'), null_bank_mode
        self.null_bank_mode = null_bank_mode
        # Softmax temperature for the NULL (unconditional) bank only: d2 -> d2/T
        # before the weights are taken. T=1 is the exact Bayes posterior over the
        # empirical prior -- which in 2048-d feature space saturates to a single
        # image (measured ESS ~1.1 of 2048 at sigma=0.92), so v*_uncond is a
        # nearest-neighbour lookup rather than a posterior MEAN. Raising T restores a
        # genuine average: T=100 measures ESS ~1022 while keeping ||v*_c - v*_u|| at
        # 0.94x its T=1 value and v*_uncond ~1.0 bank-mean-norms from the plain bank
        # mean. T >= 1000 degenerates to the bank mean itself (dist 0.06).
        #
        # Deliberately NOT applied to the conditional bank: v*_cond is the
        # class-restricted target the emp_fm term regresses onto, and smoothing it
        # would blur the class identity the CFG gap is supposed to isolate.
        self.null_temp = float(null_temp)
        # Space the posterior WEIGHTS are scored in. The weighted SUM is always over
        # the full latents; only the softmax distances differ.
        #   'feat'  -- the rank-8 subspace features (2048-d). Cheap, and at T=1 the
        #              lesser evil: the log-weight spread grows ~sqrt(dims), so full-D
        #              at T=1 is an even harder argmin. But it computes E[x0 | P x_t]:
        #              the complement of x0_hat is averaged over images chosen without
        #              ever consulting the complement.
        #   'latent' -- the full 196608-d diffusion state, i.e. the same space the
        #              on-path x_0 / x_t live in (patchify(vae.encode(img*2-1)); with
        #              the identity RGBColorEncoder that is also pixel space). The
        #              exact Gaussian posterior over the bank,
        #              E[x0 | x_t]. Only sensible WITH temp_spread, which normalises the
        #              dimension-driven spread away; at T=1 it degenerates to argmin.
        #              Costs a second streamed pass over the bank latents per row.
        assert kernel_space in ('feat', 'latent'), kernel_space
        self.kernel_space = kernel_space
        # Optional preprocessed uint8 cache prefix (tools/build_imagenet_u8_cache.py).
        # Bit-identical to decoding the JPEGs (same image_preproc, flips applied later
        # exactly as now), so it changes speed, not the banks. None -> JPEG path.
        from lakonlab.datasets.u8_cache import U8ImageCache
        self._u8 = U8ImageCache(u8_cache) if u8_cache else None
        # ADAPTIVE temperature, applied to BOTH banks: divide d2 by T so that the
        # post-scaling spread sd(d2/T) equals temp_spread. T is read off the data --
        # T = sd(d2)/temp_spread -- so it tracks sigma, bank size and feature scale
        # without a hard-coded constant.
        #
        # Why adaptive rather than the fixed null_temp: the raw spread varies ~12x
        # across the band (measured sd 10.2 at sigma=0.98, 45 at 0.95, 119 at 0.92,
        # 287 at 0.88), because it is (1-s)^2/(2 s^2) * sd(||Dj||^2) with
        # sd(||Dj||^2) ~ 31,600 roughly constant. A single T therefore over-smooths
        # one end of the band and under-smooths the other: T=100 leaves spread 1.19
        # at sigma=0.92 but 0.10 at 0.98, i.e. essentially the plain bank mean there.
        #
        # Clamped to T >= 1 so this only ever SMOOTHS; where the softmax is already
        # diffuse enough it is left alone rather than sharpened.
        #
        # Unlike null_temp this applies to the conditional bank too, so v*_cond
        # becomes a class-restricted posterior MEAN rather than a nearest-neighbour
        # lookup. Takes precedence over null_temp when set.
        self.temp_spread = None if temp_spread is None else float(temp_spread)
        # Restrict the adaptive temperature to the NULL bank, leaving v*_cond the
        # exact (saturated) posterior. Isolates "smooth the unconditional branch"
        # from "smooth both".
        self.temp_spread_null_only = bool(temp_spread_null_only)
        # A SEPARATE temp_spread for the null bank (None -> temp_spread, as before).
        # The two banks' MSE-optimal smoothing differs: a 1300-image class bank and a
        # 10k all-class bank saturate differently, and measured (held-out images,
        # tools/vstar_cond_probe.py) the null optimum sits at a lower temperature
        # (larger spread) than the conditional one.
        self.temp_spread_null = None if temp_spread_null is None else float(temp_spread_null)
        self.random_flip = random_flip   # match the training augmentation (h-flip p=0.5)
        self.include_flips = include_flips  # add BOTH h-orientations of every image to the bank
        self.num_workers = num_workers
        self.sample_chunk = sample_chunk
        self.bank_chunk = bank_chunk
        self.min_sigma = min_sigma
        self._pool = None
        self._proc_pool = None  # lazy per-process thread pool for image loading

        # class -> [rel paths]  (+ flat list for the unconditional draw)
        self._class_paths = [[] for _ in range(num_classes)]
        self._all_paths = []
        text = FileClient.infer_client(uri=datalist_path).get_text(datalist_path)
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) < 2:
                continue
            path, lab = parts[0], int(parts[1])
            self._all_paths.append(path)
            if 0 <= lab < num_classes:
                self._class_paths[lab].append(path)
        # empirical class prior (true dataset counts) for proportional sampling
        self._class_counts = torch.tensor(
            [len(p) for p in self._class_paths], dtype=torch.long)
        self._file_client = None  # lazy, per-process (fork-safe)

    @property
    def file_client(self):
        if self._file_client is None:
            self._file_client = FileClient.infer_client(uri=self.data_root)
        return self._file_client

    @property
    def ready(self):
        # data lives on disk -> ready as soon as the datalist is parsed.
        return int(self._class_counts.sum()) > 0

    @torch.no_grad()
    def sample_labels(self, num, device):
        """Sample ``num`` class labels proportional to the dataset's per-class image
        counts (the true class prior); matches the on-path stream's distribution."""
        counts = self._class_counts.float()
        total = float(counts.sum())
        if total <= 0:
            return torch.randint(0, self.num_classes, (num,), device=device)
        return torch.multinomial(counts / total, num, replacement=True).to(device)

    def _load_one(self, p):
        data_path = self.file_client.join_path(self.data_root, p)
        data_bytes = self.file_client.get(data_path)
        img = Image.open(BytesIO(data_bytes)).convert('RGB')
        arr = image_preproc(img, self.image_size, random_flip=self.random_flip)
        return torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

    def _load_images(self, paths, device):
        """Load + center-crop a list of images to ``[B, 3, H, W]`` in ``[0, 1]``
        (identical preprocessing to the ImageNet dataset), in parallel."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.num_workers)
        imgs = list(self._pool.map(self._load_one, paths))
        return torch.stack(imgs).to(device)

    def _draw_paths(self, class_id):
        """Paths for a class's bank. Real class: ``bank_size`` images (``None`` ->
        the entire class). Null class: from the whole dataset, always capped at
        ``null_bank_size``."""
        if 0 <= class_id < self.num_classes and len(self._class_paths[class_id]) > 0:
            pool = self._class_paths[class_id]
            cap = self.bank_size            # None -> entire class
        else:
            pool = self._all_paths          # null / unconditional -> global draw
            cap = self.null_bank_size       # never load the whole dataset
        n = len(pool) if cap is None else min(cap, len(pool))
        idx = torch.randperm(len(pool))[:n].tolist()
        return [pool[i] for i in idx]

    @torch.no_grad()
    def _load_images_u8(self, paths):
        """IO half of a bank: decode ``paths`` to a uint8 CPU tensor ``[M, H, W, 3]``.

        Touches no CUDA and returns CPU memory, so it can run on a prefetch thread
        while the GPU is busy. uint8 HWC keeps the in-flight buffer a quarter of what
        float CHW would cost -- the flip, the float conversion and the encode all
        happen on the GPU when the bank is finished.
        """
        # THREADS, deliberately. Processes measure 2.7x faster on decode (3.5k ->
        # 9.5k img/s) because PIL holds the GIL, but neither pool survives here: a
        # FORK context deadlocks against the parent's initialised torch/CUDA (observed
        # hanging indefinitely), and a SPAWN context died with BrokenProcessPool under
        # the node's training load. A loader that can hang a DDP run is worse than a
        # slower one, so the speedup is left on the table; the prefetch in
        # GaussianFlowOnPolicy._prefetch_banks hides most of this cost anyway.
        if self._u8 is not None:        # preprocessed cache: threaded pread, no decode
            out = np.empty((len(paths), ) + self._u8.image_shape, np.uint8)
            return torch.from_numpy(self._u8.read_into(paths, out, self.u8_read_threads))
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.num_workers)
        args = [(self.data_root, rp, self.image_size) for rp in paths]
        return torch.stack(list(self._pool.map(_load_one_u8_worker, args)))

    def _load_one_u8(self, rel_path):
        img = Image.open(self.file_client.join_path(self.data_root, rel_path)).convert('RGB')
        return torch.from_numpy(image_preproc(img, self.image_size, random_flip=False))

    @torch.no_grad()
    def _finish_bank(self, imgs_u8, encode_fn, feat_fn, device, chunk=1024, lease=None):
        """GPU half of a bank: ``uint8 CPU [M,H,W,3]`` -> a :class:`U8Bank`
        (``bank_storage='u8'``) or a ``(latents_cpu bf16, feats)`` tuple ('latent').
        include_flips adds both h-orientations (free, in-memory), laid out as
        ``[originals; flips]``. ``chunk`` entries are encoded at a time, across
        orientations -- each uint8 chunk is uploaded once and flipped on the GPU --
        so a large shared null bank (20k images is ~31 GB as fp32 with flips) never
        sits on the GPU whole; encode_fn and feat_fn act per image, so chunking
        changes nothing."""
        if lease is not None:
            lease[0].check(lease[1])
        orients = (False, True) if self.include_flips else (False, )
        keep_u8 = self.bank_storage == 'u8'
        lats = [[] for _ in orients]
        feats = [[] for _ in orients]
        step = max(1, chunk // len(orients))
        for b0 in range(0, imgs_u8.shape[0], step):
            imgs = imgs_u8[b0:b0 + step].to(device, non_blocking=True).permute(
                0, 3, 1, 2).float() / 255.0
            for o, flipped in enumerate(orients):
                x = torch.flip(imgs, dims=[-1]) if flipped else imgs
                latents = encode_fn(x)                        # [m, C, H, W] diffusion input space
                feats[o].append(feat_fn(latents).flatten(1))  # [m, Df]
                if not keep_u8:
                    # bf16 latents on CPU: half the RAM; x0_hat is a weighted average
                    # so the precision loss is negligible.
                    lats[o].append(latents.detach().to(torch.bfloat16).cpu())
        feats = torch.cat([f for per_o in feats for f in per_o]).detach()
        if keep_u8:
            return U8Bank(imgs_u8, feats, encode_fn, len(orients), lease=lease)
        return torch.cat([lat for per_o in lats for lat in per_o]), feats

    @torch.no_grad()
    def _build_one(self, paths, encode_fn, feat_fn, device):
        """Synchronous build (IO then GPU), kept for callers that do not prefetch."""
        return self._finish_bank(self._load_images_u8(paths), encode_fn, feat_fn, device)

    def draw_bank_paths(self, labels, null_classes=None):
        """The path draws for one bank build, WITHOUT loading anything.

        Returns the opaque payload consumed by :meth:`load_bank_images` and then
        :meth:`finish_banks`.

        The CONDITIONAL side is DEDUPLICATED BY CLASS when the draw is deterministic.
        With ``bank_size=None`` ``_draw_paths`` returns the entire class (``randperm``
        only permutes it), and ``x0_hat`` consumes a bank as a SET -- a softmax-weighted
        average over its rows -- so two trajectories of one class compute an identical
        ``x0_hat`` from identical images. Loading and encoding that class once and
        sharing it by reference is therefore exact, not an approximation. It only pays
        off alongside ``band_classes_per_batch``: under the default 'prior' sampling
        over 1000 classes a 21-row draw is almost all distinct and there is nothing to
        merge.

        With ``bank_size`` SET the draw is a random subset per call, so same-class rows
        are genuinely different banks and dedup would change the statistics -- it is
        skipped.

        The NULL draw is INDEPENDENT per trajectory by default (a fresh
        ``null_bank_size`` sample for every row), so trajectories keep distinct
        unconditional posteriors. ``null_bank_mode='shared'`` draws it ONCE and points
        every trajectory at it (``null_index``), the same by-reference sharing as the
        conditional dedup.
        """
        labels = labels.tolist() if torch.is_tensor(labels) else [int(x) for x in labels]
        labels = [int(c) for c in labels]
        if self.bank_size is None:
            uniq = list(dict.fromkeys(labels))        # distinct classes, order kept
            slot = {c: i for i, c in enumerate(uniq)}
            cond_index = [slot[c] for c in labels]
        else:                                          # random subsets -> no dedup
            uniq = labels
            cond_index = list(range(len(labels)))
        n_null = 1 if self.null_bank_mode == 'shared' else len(labels)
        return dict(
            cond_paths=[self._draw_paths(c) for c in uniq],
            cond_index=cond_index,
            # null_classes restricts the UNCONDITIONAL draw to a given set of classes
            # instead of the whole dataset. Off by default: v*_uncond is the target for
            # the model's unconditional branch, whose null embedding is trained on the
            # FULL marginal, so a restricted null bank makes target and prediction refer
            # to different quantities (measured: SNR 15.8 -> 7.3, and the reference then
            # moves every iteration with the batch's class draw). Provided so that
            # choice can be measured rather than assumed.
            null_paths=[self._draw_paths_restricted(null_classes)
                        if null_classes is not None else self._draw_paths(-1)
                        for _ in range(n_null)],
            null_index=[0] * len(labels) if n_null == 1 else list(range(len(labels))))

    def lease_host_buffer(self, drawn):
        """MAIN THREAD. Lease a reusable host buffer sized for ``drawn``'s banks and
        carve it into one exact view per distinct bank. Returns the lease, to pass to
        :meth:`load_bank_images` and later to :meth:`release_host_buffer` -- or None
        (no u8 cache, reuse disabled, or every buffer still busy), in which case the
        load allocates fresh memory exactly as before. Never hands out a buffer that
        is in use."""
        sizes = [len(p) for p in drawn['cond_paths']] + [len(p) for p in drawn['null_paths']]
        if self._u8 is None or self.host_buffers <= 0 or sum(sizes) == 0:
            return None
        buf = next((b for b in self._host_bufs if not b.busy), None)
        if buf is None:
            if len(self._host_bufs) >= self.host_buffers:
                return None
            buf = HostBankBuffer(self._u8.image_shape, self.pin_host_buffers)
            self._host_bufs.append(buf)
        gen = buf.lease(sum(sizes))
        views, off = [], 0
        for n in sizes:
            views.append(buf.tensor[off:off + n])
            off += n
        nc = len(drawn['cond_paths'])
        return dict(buf=buf, gen=gen, cond=views[:nc], null=views[nc:])

    @staticmethod
    def release_host_buffer(lease):
        """Return a lease's buffer to the pool. The caller guarantees nothing reads
        it any more: for a consumed band, the band is over and the device synced;
        for an abandoned draw, its load has finished. Safe from any thread (it only
        flips a flag; no CUDA call)."""
        if lease is not None:
            lease['buf'].release(lease['gen'])

    @property
    def host_buffer_bytes(self):
        """Host bytes held by the reusable bank buffers (pinned or not)."""
        return sum(b.nbytes for b in self._host_bufs)

    @torch.no_grad()
    def load_bank_images(self, drawn, lease=None):
        """IO for one :meth:`draw_bank_paths` payload -- the whole prefetch payload,
        CPU only. Loads each DISTINCT conditional and null bank once: into ``lease``'s
        views of a reused buffer when given (see :meth:`lease_host_buffer`), else into
        fresh memory."""
        if lease is None:
            cond_u8 = [self._load_images_u8(pth) for pth in drawn['cond_paths']]
            null_u8 = [self._load_images_u8(pth) for pth in drawn['null_paths']]
        else:
            lease['buf'].check(lease['gen'])
            for pth, view in zip(drawn['cond_paths'] + drawn['null_paths'],
                                 lease['cond'] + lease['null']):
                self._u8.read_into(pth, view.numpy(), self.u8_read_threads)
            cond_u8, null_u8 = lease['cond'], lease['null']
        return dict(
            cond_u8=cond_u8,
            cond_index=drawn['cond_index'],
            null_u8=null_u8,
            null_index=drawn['null_index'],
            lease=lease)

    @torch.no_grad()
    def finish_banks(self, loaded, encode_fn, feat_fn, device):
        """GPU half for a prefetched payload -> ``(cond_banks, null_banks)``, both
        indexed BY TRAJECTORY as ``x0_hat`` expects.

        Each distinct bank is finished once and the result (see :meth:`_finish_bank`)
        is shared by reference across the rows that use it (a class's rows, or every row
        under null_bank_mode='shared'); ``x0_hat`` only reads them, so the aliasing is
        safe and saves the RAM as well as the IO.
        """
        lease = loaded.get('lease')
        lease = None if lease is None else (lease['buf'], lease['gen'])
        cond_uniq = [self._finish_bank(u8, encode_fn, feat_fn, device, lease=lease)
                     for u8 in loaded['cond_u8']]
        cond_banks = [cond_uniq[i] for i in loaded['cond_index']]
        null_uniq = [self._finish_bank(u8, encode_fn, feat_fn, device, lease=lease)
                     for u8 in loaded['null_u8']]
        null_banks = [null_uniq[i] for i in loaded['null_index']]
        return cond_banks, null_banks

    @torch.no_grad()
    def build_banks(self, labels, encode_fn, feat_fn, device, null_classes=None):
        """Synchronous build for callers that do not prefetch (the DaggerRolloutHook).

        Routed through draw -> load -> finish so it gets the same per-class dedup;
        returns ``(cond_banks, null_banks)``, each a list of banks (see
        :meth:`_finish_bank`) indexed by trajectory position.

        ``encode_fn`` maps loaded images ``[M, 3, H, W] in [0,1]`` to the diffusion
        input space (``patchify(vae.encode(img*2-1))``); ``feat_fn`` -> compact subspace.
        """
        drawn = self.draw_bank_paths(labels, null_classes=null_classes)
        return self.finish_banks(
            self.load_bank_images(drawn), encode_fn, feat_fn, device)

    def _draw_paths_restricted(self, classes):
        """null_bank_size paths drawn from the union of ``classes`` only."""
        pool = []
        for c in set(int(x) for x in classes):
            if 0 <= c < self.num_classes:
                pool.extend(self._class_paths[c])
        if not pool:
            return self._draw_paths(-1)
        n = min(self.null_bank_size, len(pool)) if self.null_bank_size else len(pool)
        idx = torch.randperm(len(pool))[:n].tolist()
        return [pool[i] for i in idx]

    @torch.no_grad()
    def x0_hat(self, x_t, sigma, feat_fn, labels, cond_banks, null_banks, null_label,
               exclude=None):
        """Posterior-mean data estimate, each row over its OWN trajectory bank:
        row ``i`` uses ``cond_banks[i]`` if ``labels[i] != null_label`` else
        ``null_banks[i]`` (both from :meth:`build_banks`, indexed by position).

        Args:
            x_t (Tensor): ``[B, C, H, W]`` visited states.
            sigma (float | Tensor): scalar or ``[B]`` noise levels in ``[0, 1]``.
            feat_fn (callable): ``[*, C, H, W] -> [*, Df]`` compact projection.
            labels (Tensor): ``[B]`` per-point label (null_label -> null bank).
            cond_banks, null_banks (list): per-trajectory banks -- :class:`U8Bank`,
                or a ``(latents_cpu, feats)`` tuple (bank_storage='latent', or built
                by hand as tools/bank_size_sweep.py does).
            exclude (list | None): per row, ``None`` or the indices of entries of the
                bank THAT ROW READS to leave out of its posterior (weight 0) -- e.g.
                both orientations of the image an on-path state was noised from.
        Returns:
            Tensor: ``[B, C, H, W]`` posterior-mean data ``x0_hat``.
        """
        B = x_t.shape[0]
        device = x_t.device
        if not torch.is_tensor(sigma):
            sigma = torch.full((B,), float(sigma), device=device)
        sigma = sigma.to(device).reshape(B)
        x_feat_all = feat_fn(x_t).flatten(1) if self.kernel_space == 'feat' else None  # [B, Df]
        labels_cpu = labels.to('cpu').long()
        out = torch.empty_like(x_t)

        # Rows grouped by the bank OBJECT they read, so a bank several rows share (the
        # class dedup, null_bank_mode='shared') crosses to the GPU once per group
        # rather than once per row. Per row the arithmetic is unchanged: the same
        # distances, temperature, softmax and chunked weighted sum, in the same order.
        groups = {}
        for i in range(B):
            is_null = int(labels_cpu[i]) == null_label
            bank = null_banks[i] if is_null else cond_banks[i]
            groups.setdefault(id(bank), (bank, []))[1].append((i, is_null))

        for bank, rows in groups.values():
            feat_m = bank.feats if isinstance(bank, U8Bank) else bank[1]
            M = feat_m.shape[0]
            scs = [sigma[i].clamp_min(self.min_sigma) for i, _ in rows]   # scalars
            if self.kernel_space == 'latent':
                # pass 1 of 2: distances on the FULL state, streamed in chunks (the
                # adaptive temperature needs sd over the whole bank before any
                # weight can be formed, so this cannot fuse with the sum below).
                d2s = [torch.empty(M, device=device) for _ in rows]
                for lo, hi, full in self._bank_chunks(bank, device):
                    for (i, _), sc, d2 in zip(rows, scs, d2s):
                        d2[lo:hi] = (x_t[i].float().unsqueeze(0) - (1 - sc) * full
                                     ).pow(2).flatten(1).sum(-1)
                d2s = [d2 / (2.0 * sc ** 2) for d2, sc in zip(d2s, scs)]
            else:
                d2s = []
                for (i, _), sc in zip(rows, scs):
                    resid = x_feat_all[i].unsqueeze(0) - (1 - sc) * feat_m   # [M, Df]
                    d2s.append(resid.pow(2).sum(-1) / (2.0 * sc ** 2))      # [M]
            ws = []
            for (i, is_null), d2 in zip(rows, d2s):
                ex = None if exclude is None else exclude[i]
                kept = d2
                if ex is not None:
                    # leave-one-out: the dropped entries get weight exactly 0, and the
                    # adaptive temperature is read off the KEPT entries only
                    keep = torch.ones(M, dtype=torch.bool, device=device)
                    keep[torch.as_tensor(ex, device=device)] = False
                    kept = d2[keep]
                    d2 = d2.masked_fill(~keep, float('inf'))
                ts = self.temp_spread_null if (is_null and self.temp_spread_null is not None) \
                    else self.temp_spread
                if (ts is not None and kept.numel() > 1
                        and (is_null or not self.temp_spread_null_only)):
                    # adaptive: rescale so sd(d2) == temp_spread
                    d2 = d2 / (kept.std() / ts).clamp_min(1.0)
                elif is_null and self.null_temp != 1.0:
                    d2 = d2 / self.null_temp              # unconditional branch only
                ws.append(torch.softmax(-d2, dim=0))      # [M]
            accs = [torch.zeros(x_t.shape[1:], device=device, dtype=out.dtype) for _ in rows]
            for lo, hi, full in self._bank_chunks(bank, device):   # full: [blk, C, H, W]
                for w, acc in zip(ws, accs):
                    acc += (w[lo:hi].view(-1, *([1] * (x_t.dim() - 1))) * full).sum(0)
            for (i, _), acc in zip(rows, accs):
                out[i] = acc

        return out

    def _bank_chunks(self, bank, device):
        """Yield ``(lo, hi, latents)`` over a bank's entries, fp32 on ``device``, in
        ``bank_chunk`` pieces -- for a :class:`U8Bank` or a legacy
        ``(latents_cpu, feats)`` tuple alike."""
        if isinstance(bank, U8Bank):
            yield from bank.chunks(device, self.bank_chunk)
            return
        lat_cpu = bank[0]
        M = lat_cpu.shape[0]
        for b0 in range(0, M, self.bank_chunk):
            b1 = min(b0 + self.bank_chunk, M)
            yield b0, b1, lat_cpu[b0:b1].to(device).float()
