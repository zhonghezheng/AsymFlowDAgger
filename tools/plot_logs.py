"""Plot metrics straight out of LakonLab run logs (direct_outputs/*.log).

Two log kinds are recognised, and either may be passed on the same command line:

  TRAINING logs (tools/train.py) -- the 'Iter(val) [n]' lines carry the periodic
    10k-image eval. Plotted as curves against the iteration, one colour per run.

  WSWEEP eval logs (tools/test.py with the wsweep config) -- one
    'val_<tag>_heun_g<w>_..._<metric> = <value>' line per guidance setting, no
    iteration axis. Plotted as FID (and friends) against the guidance scale w,
    one line per run.

Usage:
    python tools/plot_logs.py <log> [<log> ...] [-o plots/] [--label NAME ...]

Each log gets a label from its filename unless --label is given (repeat --label
once per log, in order). Writes plots/train_curves.png and/or plots/wsweep_fid.png
depending on which kinds were passed.
"""

import argparse
import os
import os.path as osp
import re
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 'Iter(val) [500]\tval_<prefix>_fid: 4.3444, val_<prefix>_precision: 0.78, ...'
VAL_RE = re.compile(r'Iter\(val\) \[(\d+)\]\s*(.*)')
# 'val_uncond_heun_g0.0_step50_fid = 27.64'  (tools/test.py, one line per metric)
TEST_RE = re.compile(r'val_(\S+?)_heun_g([0-9.]+)\S*?_(fid|precision|recall|is) = ([0-9.eE+-]+)')
# eval keys carry the sampler/guidance tag, so they hold '.', '(', ')' and '-'
# as well as word chars: 'val_heun_g2.3(0-0.88)_step50_fid: 4.3444'.
KV_RE = re.compile(r'([\w.()\-]+):\s*([0-9.eE+-]+)')
# strip the sampler/guidance prefix so 'val_heun_g2.3(0-0.88)_step50_fid' -> 'fid'
VAL_KEY_RE = re.compile(r'^val_.*?_(fid|precision|recall|is)$')

METRICS = ['fid', 'is', 'precision', 'recall']
TITLES = dict(fid='FID (10k)', **{'is': 'Inception Score'},
              precision='Precision', recall='Recall')
# which direction is an improvement -- FID is a distance (lower is better), the
# rest are scores (higher is better). Shown as an arrow in each panel title.
LOWER_IS_BETTER = {'fid'}


def panel_title(key):
    arrow = 'v lower' if key in LOWER_IS_BETTER else '^ higher'
    arrow = arrow.replace('v', '\u2193').replace('^', '\u2191')
    return f'{TITLES.get(key, key)}  ({arrow} is better)'


def maybe_log(ax, values):
    """Log-scale only when the spread earns it -- FID spans 4.3-32 across a
    guidance sweep (log helps) but only 4.35-4.7 along a training run (log just
    turns the ticks into '4.7 x 10^0')."""
    values = [v for v in values if v and v > 0]
    if values and max(values) / min(values) > 4:
        ax.set_yscale('log')


def parse(path):
    """Return dict(kind='train'|'wsweep', ...) of everything found in one log."""
    val, sweep = OrderedDict(), OrderedDict()

    with open(path, errors='replace') as f:
        for line in f:
            m = TEST_RE.search(line)
            if m:
                tag, w, metric, value = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
                sweep.setdefault(w, {'tag': tag})[metric] = value
                continue

            m = VAL_RE.search(line)
            if m:
                it = int(m.group(1))
                for key, value in KV_RE.findall(m.group(2)):
                    short = VAL_KEY_RE.match(key)
                    if short:
                        val.setdefault(short.group(1), []).append((it, float(value)))

    kind = 'wsweep' if sweep and not val else 'train'
    return dict(kind=kind, path=path, val=val,
                sweep=OrderedDict(sorted(sweep.items())))


def default_label(path):
    """'out_log_asymflow_h_16_r8_imagenet_dagger_full_bank64_4gpus_2026...' -> 'dagger_full_bank64'."""
    stem = osp.splitext(osp.basename(path))[0]
    stem = re.sub(r'^out_log_', '', stem)
    stem = re.sub(r'_\d{8}_\d{6}$', '', stem)
    stem = stem.replace('asymflow_h_16_r8_imagenet_', '').replace('_4gpus', '')
    return re.sub(r'^wsweep_', '', stem) or stem


def _series(pairs):
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def plot_train(runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for ax, key in zip(axes.flat, METRICS):
        seen = []
        for run in runs:
            pairs = run['val'].get(key)
            if not pairs:
                continue
            xs, ys = _series(pairs)
            seen += ys
            ax.plot(xs, ys, marker='o', ms=3.5, lw=1.6, alpha=0.9,
                    color=run['color'], label=run['label'])
        ax.set_title(panel_title(key))
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.25)
        maybe_log(ax, seen)
        if ax is axes.flat[0]:
            ax.legend(fontsize=9)

    fig.suptitle('Training runs -- 10k-image eval '
                 '(Heun step-50, g=2.3, interval [0, 0.88])', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_wsweep(runs, out_path, cfg_only=False):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    for ax, key in zip(axes, METRICS):
        seen = []
        for run in runs:
            ws = list(run['sweep'].keys())
            ys = [run['sweep'][w].get(key) for w in ws]
            seen += ys
            ax.plot(ws, ys, marker='o', ms=6, lw=1.8, color=run['color'], label=run['label'])
            # Annotate only the points worth reading: the two reference settings
            # (uncond / cond) and the run's OPTIMUM. A dense CFG grid annotated at
            # every point just overprints itself, and the axis already carries the
            # trend. The optimum gets a ring so it is findable at a glance.
            best = (min if key in LOWER_IS_BETTER else max)(
                range(len(ys)), key=lambda i: ys[i] if ys[i] is not None else
                (float('inf') if key in LOWER_IS_BETTER else float('-inf')))
            ax.plot([ws[best]], [ys[best]], marker='o', ms=12, mfc='none',
                    mec=run['color'], mew=1.6, zorder=5)
            for i, (w, y) in enumerate(zip(ws, ys)):
                if w not in (0.0, 1.0) and i != best:
                    continue
                others = [r['sweep'][w].get(key) for r in runs
                          if r is not run and w in r['sweep']]
                above = not others or y >= max(
                    (o for o in others if o is not None), default=y)
                ax.annotate(f'{y:.3g}', (w, y), textcoords='offset points',
                            xytext=(0, 11 if above else -18), ha='center',
                            fontsize=8, fontweight='bold', color=run['color'])
        ax.set_title(panel_title(key))
        ax.set_xlabel('guidance scale $w$')
        ax.grid(alpha=0.25)
        ax.margins(y=0.16)
        maybe_log(ax, seen)
        # label what the three sampled w values actually mean
        # only 0 and 1 get a word; the CFG grid is dense enough that repeating
        # 'CFG' under every tick just collides with its neighbours.
        ws_all = sorted({w for run in runs for w in run['sweep']})
        ax.set_xticks(ws_all)
        ax.set_xticklabels(
            [{0.0: '0\nuncond', 1.0: '1\ncond'}.get(w, f'{w:g}') for w in ws_all],
            fontsize=8)
        # runs on different CFG grids put ticks within ~0.1 of each other (e.g. a
        # 2.3-only run beside a 2.2/2.4 one); rotate just those so they stay legible.
        for lbl, w in zip(ax.get_xticklabels(), ws_all):
            if w not in (0.0, 1.0):
                lbl.set_rotation(45)
                lbl.set_ha('right')
        if ax is axes[0]:
            ax.legend(fontsize=9)

    ws_all = sorted({w for run in runs for w in run['sweep']})
    span = f'{min(ws_all):g}-{max(ws_all):g}' if ws_all else 'n/a'
    head = (f'CFG guidance sweep at iter 5000 (w={span}, interval [0, 0.88])'
            if cfg_only else
            f'Guidance sweep at iter 5000 -- unconditional (w=0), conditional (w=1), '
            f'CFG (w={span}, interval [0, 0.88])')
    fig.suptitle(f'{head}; 10k images, Heun step-50.  '
                 "Ringed marker = each run's optimum.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+', help='run logs (training and/or wsweep eval)')
    parser.add_argument('-o', '--out-dir', default='plots', help='output directory')
    parser.add_argument('--cfg-only', action='store_true',
                        help='sweep plots: drop the w=0 (uncond) and w=1 (cond) '
                             'reference points and show only the CFG region')
    parser.add_argument('--label', action='append', default=[],
                        help='label for each log, in order (default: derived from filename)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs, colors = [], {}
    for i, path in enumerate(args.logs):
        run = parse(path)
        run['label'] = args.label[i] if i < len(args.label) else default_label(path)
        # one colour per LABEL, not per position: the same run usually appears in
        # both figures (training curve + guidance sweep) at different indices.
        run['color'] = colors.setdefault(run['label'], f'C{len(colors)}')
        runs.append(run)
        n = len(run['sweep']) if run['kind'] == 'wsweep' else len(run['val'].get('fid', []))
        print(f"{run['label']:28s} kind={run['kind']:7s} points={n:3d}  {path}")

    written = []
    train_runs = [r for r in runs if r['kind'] == 'train']
    sweep_runs = [r for r in runs if r['kind'] == 'wsweep']
    if train_runs:
        written.append(plot_train(train_runs, osp.join(args.out_dir, 'train_curves.png')))
    if sweep_runs:
        if args.cfg_only:
            for r in sweep_runs:
                r['sweep'] = OrderedDict(
                    (w, v) for w, v in r['sweep'].items() if w not in (0.0, 1.0))
            sweep_runs = [r for r in sweep_runs if r['sweep']]
        written.append(plot_wsweep(sweep_runs, osp.join(args.out_dir, 'wsweep_fid.png'),
                                   cfg_only=args.cfg_only))

    for path in written:
        print('wrote', path)


if __name__ == '__main__':
    main()
