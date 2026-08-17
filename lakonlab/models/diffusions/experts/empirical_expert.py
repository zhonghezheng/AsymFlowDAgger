# Copyright (c) 2026 Hansheng Chen

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from PIL import Image
from mmcv.fileio import FileClient

from ...builder import MODULES
from lakonlab.datasets.imagenet import image_preproc


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
        num_workers (int): threads for the on-disk image load/decode (GIL-released).
        sample_chunk (int): rows processed at once in x0_hat (bounds memory).
        bank_chunk (int): bank entries summed at once (bounds peak memory).
        min_sigma (float): floor on ``sigma`` in the posterior denominator.
    """

    def __init__(self,
                 datalist_path,
                 data_root,
                 image_size=256,
                 num_classes=1000,
                 bank_size=256,
                 null_bank_size=512,
                 random_flip=True,
                 include_flips=False,
                 num_workers=32,
                 sample_chunk=32,
                 bank_chunk=128,
                 min_sigma=1e-3):
        super().__init__()
        self.datalist_path = datalist_path
        self.data_root = data_root
        self.image_size = image_size
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.null_bank_size = null_bank_size
        self.random_flip = random_flip   # match the training augmentation (h-flip p=0.5)
        self.include_flips = include_flips  # add BOTH h-orientations of every image to the bank
        self.num_workers = num_workers
        self.sample_chunk = sample_chunk
        self.bank_chunk = bank_chunk
        self.min_sigma = min_sigma
        self._pool = None  # lazy per-process thread pool for image loading

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
    def _build_one(self, paths, encode_fn, feat_fn, device):
        """Load one bank from ``paths`` -> ``(latents_cpu bf16, feats)``. include_flips
        adds both h-orientations (free, in-memory); bf16 halves the CPU latent RAM."""
        imgs = self._load_images(paths, device)
        if self.include_flips:
            imgs = torch.cat([imgs, torch.flip(imgs, dims=[-1])], dim=0)
        latents = encode_fn(imgs)                 # [M, C, H, W] diffusion input space
        feats = feat_fn(latents).flatten(1)       # [M, Df]
        # bf16 latents on CPU: half the RAM; x0_hat is a weighted average so the
        # precision loss is negligible.
        return (latents.detach().to(torch.bfloat16).cpu(), feats.detach())

    @torch.no_grad()
    def build_banks(self, labels, encode_fn, feat_fn, device):
        """Build one INDEPENDENT bank PER TRAJECTORY. For each row ``i`` (sampled
        class ``labels[i]``): a conditional bank (``bank_size`` draw from that class)
        AND a null bank (``null_bank_size`` draw from the whole dataset). Returns
        ``(cond_banks, null_banks)``, each a list of ``(latents_cpu, feats)`` indexed
        by trajectory position -- no per-class or shared-null pooling, so different
        trajectories (even of the same class) get different posteriors.

        ``encode_fn`` maps loaded images ``[M, 3, H, W] in [0,1]`` to the diffusion
        input space (``patchify(vae.encode(img*2-1))``); ``feat_fn`` -> compact subspace.
        """
        labels = labels.tolist() if torch.is_tensor(labels) else [int(x) for x in labels]
        cond_banks, null_banks = [], []
        for c in labels:
            cond_banks.append(self._build_one(self._draw_paths(int(c)), encode_fn, feat_fn, device))
            null_banks.append(self._build_one(self._draw_paths(-1), encode_fn, feat_fn, device))  # -1 -> null draw
        return cond_banks, null_banks

    @torch.no_grad()
    def x0_hat(self, x_t, sigma, feat_fn, labels, cond_banks, null_banks, null_label):
        """Posterior-mean data estimate, each row over its OWN trajectory bank:
        row ``i`` uses ``cond_banks[i]`` if ``labels[i] != null_label`` else
        ``null_banks[i]`` (both from :meth:`build_banks`, indexed by position).

        Args:
            x_t (Tensor): ``[B, C, H, W]`` visited states.
            sigma (float | Tensor): scalar or ``[B]`` noise levels in ``[0, 1]``.
            feat_fn (callable): ``[*, C, H, W] -> [*, Df]`` compact projection.
            labels (Tensor): ``[B]`` per-point label (null_label -> null bank).
            cond_banks, null_banks (list): per-trajectory ``(latents_cpu, feats)``.
        Returns:
            Tensor: ``[B, C, H, W]`` posterior-mean data ``x0_hat``.
        """
        B = x_t.shape[0]
        device = x_t.device
        if not torch.is_tensor(sigma):
            sigma = torch.full((B,), float(sigma), device=device)
        sigma = sigma.to(device).reshape(B)
        x_feat_all = feat_fn(x_t).flatten(1)  # [B, Df]
        labels_cpu = labels.to('cpu').long()
        out = torch.empty_like(x_t)

        for i in range(B):
            lat_cpu, feat_m = (null_banks[i] if int(labels_cpu[i]) == null_label
                               else cond_banks[i])
            M = feat_m.shape[0]
            sc = sigma[i].clamp_min(self.min_sigma)       # scalar
            xf = x_feat_all[i]                            # [Df]
            resid = xf.unsqueeze(0) - (1 - sc) * feat_m   # [M, Df]
            d2 = resid.pow(2).sum(-1) / (2.0 * sc ** 2)   # [M]
            w = torch.softmax(-d2, dim=0)                 # [M]
            acc = torch.zeros(x_t.shape[1:], device=device, dtype=out.dtype)
            for b0 in range(0, M, self.bank_chunk):
                b1 = min(b0 + self.bank_chunk, M)
                full = lat_cpu[b0:b1].to(device).float()  # [blk, C, H, W]
                acc += (w[b0:b1].view(-1, *([1] * (x_t.dim() - 1))) * full).sum(0)
            out[i] = acc

        return out
