"""Build the preprocessed uint8 image cache read by lakonlab.datasets.u8_cache.

One pass over the datalist: every image goes through the SAME image_preproc the live
loaders use (random_flip=False; flips are applied at read time), and lands in one
``[N, S, S, 3]`` uint8 memmap sorted by (class, datalist order). Output, for a prefix
P (default data/imagenet/train_u8_<S>):

    P.u8.npy       the images, N * S * S * 3 bytes (252 GB for ImageNet train @ 256)
    P.index.npz    relpaths [N], labels [N], class_offsets [C+1]
    P.progress     index of the next chunk to write (resume point)
    P.complete     written LAST, only after every row is filled -- readers refuse
                   a cache without it, since unwritten rows read back as black images

Resumable: rerun the same command and it continues from P.progress. Runs outside
training, so a plain fork process pool is safe here (it is not inside DDP -- the
reason the in-training loader has to use threads).

    python tools/build_imagenet_u8_cache.py [--workers 32] [--size 256]
    python tools/build_imagenet_u8_cache.py --verify 2000   # spot-check vs JPEG
"""
import argparse, os, os.path as osp, sys, time
from multiprocessing import get_context

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from lakonlab.datasets.imagenet import image_preproc  # noqa: E402

_ROOT = _SIZE = None


def _init(root, size):
    global _ROOT, _SIZE
    _ROOT, _SIZE = root, size


def _one(rel):
    img = Image.open(osp.join(_ROOT, rel)).convert('RGB')
    return image_preproc(img, _SIZE, random_flip=False)


def read_datalist(path, num_classes):
    rel, lab = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and 0 <= int(p[1]) < num_classes:
                rel.append(p[0]); lab.append(int(p[1]))
    lab = np.asarray(lab, dtype=np.int32)
    order = np.argsort(lab, kind='stable')           # class-sorted, datalist order kept
    rel = np.asarray(rel)[order]; lab = lab[order]
    offsets = np.searchsorted(lab, np.arange(num_classes + 1)).astype(np.int64)
    return rel, lab, offsets


def build(a):
    rel, lab, offsets = read_datalist(a.datalist, a.num_classes)
    n, S = len(rel), a.size
    print(f'{n:,} images, {a.num_classes} classes -> {a.prefix}.u8.npy '
          f'({n * S * S * 3 / 1e9:.1f} GB)', flush=True)
    idx_path = a.prefix + '.index.npz'
    if osp.exists(idx_path):                          # resuming: must be the same list
        old = np.load(idx_path)
        assert len(old['relpaths']) == n and (old['relpaths'] == rel).all(), \
            'datalist changed since this cache was started; delete it and rebuild'
    else:
        np.savez(idx_path, relpaths=rel, labels=lab, class_offsets=offsets)
    mode = 'r+' if osp.exists(a.prefix + '.u8.npy') else 'w+'
    mm = np.lib.format.open_memmap(a.prefix + '.u8.npy', mode=mode,
                                   dtype=np.uint8, shape=(n, S, S, 3))
    prog = a.prefix + '.progress'
    start = int(open(prog).read()) if osp.exists(prog) else 0
    chunks = list(range(0, n, a.chunk))
    print(f'resuming at chunk {start}/{len(chunks)}' if start else 'starting', flush=True)
    t0 = time.time(); done = 0
    with get_context('fork').Pool(a.workers, _init, (a.root, S)) as pool:
        for ci in range(start, len(chunks)):
            b0 = chunks[ci]; b1 = min(b0 + a.chunk, n)
            mm[b0:b1] = np.stack(pool.map(_one, rel[b0:b1], chunksize=16))
            mm.flush()
            with open(prog, 'w') as f:
                f.write(str(ci + 1))
            done += b1 - b0
            r = done / (time.time() - t0)
            print(f'  chunk {ci + 1}/{len(chunks)}  {b1:,}/{n:,}  {r:,.0f} img/s  '
                  f'eta {(n - b1) / r / 60:.1f} min', flush=True)
    del mm
    open(a.prefix + '.complete', 'w').write(f'{n} {S}\n')
    print(f'complete in {(time.time() - t0) / 60:.1f} min', flush=True)


def verify(a):
    from lakonlab.datasets.u8_cache import U8ImageCache
    c = U8ImageCache(a.prefix); c._open()
    rel = np.load(a.prefix + '.index.npz')['relpaths']
    pick = np.random.default_rng(0).choice(len(rel), size=min(a.verify, len(rel)),
                                           replace=False)
    _init(a.root, a.size)
    bad = sum(not np.array_equal(c.get([rel[i]])[0].numpy(), _one(rel[i])) for i in pick)
    print(f'verify: {len(pick) - bad}/{len(pick)} bit-identical to the JPEG path')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datalist', default='data/imagenet/train.txt')
    ap.add_argument('--root', default='data/imagenet/train/')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--num-classes', type=int, default=1000)
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--verify', type=int, default=0, help='spot-check N rows and exit')
    a = ap.parse_args()
    a.prefix = a.prefix or f'data/imagenet/train_u8_{a.size}'
    verify(a) if a.verify else build(a)
