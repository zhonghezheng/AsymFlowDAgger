"""Report-style guidance-sweep figure for the MMD arms (2x2, CFG region only).

Same data as tools/plot_logs.py, presented the way the arm-comparison figures are:
one colour per swept weight, linestyle for the band shape, the baseline as a dashed
blue reference with its best value called out, and a subtitle stating what the
panels show rather than leaving the reader to infer it.

Every panel is zoomed to the CFG region (w >= 1.8). The w=0 / w=1 reference points
are two orders of magnitude away on FID and would flatten all four panels.

    python tools/plot_wsweep_report.py            # writes plots/wsweep_mmd.png
"""

import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from plot_logs import parse  # noqa: E402

W = ('direct_outputs/out_log_asymflow_h_16_r8_imagenet_wsweep'
     '_asymflow_h_16_r8_imagenet')
RAMP = ['#fdbe85', '#e6550d', '#8c2d04']          # mmd_weight 10 / 100 / 1000
BASE = '#1f77b4'
W_MIN = 2.0                                        # CFG region lower edge
# Upper edge: only reg_ft was swept past 2.6, so keeping 2.8 would give the
# rightmost tick a single curve and stretch the axis over a range the MMD arms
# were never measured at.
W_MAX = 2.6

# colour encodes mmd_weight, linestyle encodes the band shape, so the one arm that
# moves the band is visually separated from the three that only change the weight.
ARMS = [
    ('mmd  w=10      band $\\sigma\\geq$0.875', [f'{W}_regft_mmd_sub10_4gpus_20260921_164929.log'],   RAMP[0], '-', 'o'),
    ('mmd  w=100     band $\\sigma\\geq$0.875', [f'{W}_regft_mmd_sub100_4gpus_20260921_181004.log'],  RAMP[1], '-', 'o'),
    ('mmd  w=1000    band $\\sigma\\geq$0.875', [f'{W}_regft_mmd_sub1000_4gpus_20260921_164929.log'], RAMP[2], '-', 'o'),
    ('mmd  w=100     window [0.75, 0.85]',      [f'{W}_regft_mmd_win_sub100_4gpus_20260921_180926.log'], RAMP[1], ':', 'D'),
    # The baseline's full sweep lives in slurm_outputs, split across two submissions
    # of the SAME checkpoint (regft_4gpus/latest.pth): the main grid and a high-w
    # tail. Merged here into the complete curve -- best FID 4.374 at w=2.4.
    ('reg_ft  (no MMD term)',
     ['slurm_outputs/wsweep_regft/out_log_wsweep_regft_13157386.out',
      'slurm_outputs/wsweep_hi_regft/out_log_wsweep_hi_regft_13309265.out'], BASE, '--', 's'),
]
PANELS = [('fid', 'FID $\\downarrow$ lower better', True),
          ('is', 'Inception Score $\\uparrow$', False),
          ('precision', 'Precision (fidelity) $\\uparrow$', False),
          ('recall', 'Recall (coverage) $\\uparrow$', False)]


def main():
    runs = []
    for label, paths, colour, ls, mk in ARMS:
        sweep = {}
        for p in paths:
            sweep.update(parse(p)['sweep'])
        sweep = {w: v for w, v in sorted(sweep.items()) if W_MIN <= w <= W_MAX}
        runs.append(dict(label=label, sweep=sweep, colour=colour, ls=ls, mk=mk))

    ws_axis = sorted({w for r in runs for w in r['sweep']})
    pos = {w: i for i, w in enumerate(ws_axis)}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for ax, (key, title, is_fid) in zip(axes.flat, PANELS):
        for r in runs:
            xs = [pos[w] for w in ws_axis if w in r['sweep']]
            ys = [r['sweep'][w][key] for w in ws_axis if w in r['sweep']]
            base = r['ls'] == '--'
            ax.plot(xs, ys, color=r['colour'], ls=r['ls'], marker=r['mk'], ms=5,
                    lw=2.4 if base else 1.8, zorder=5 if base else 3,
                    label=r['label'])
        if is_fid:
            # ring + label each arm's own FID optimum, so the per-arm best is
            # readable without tracing curves back to the axis
            for r in runs:
                ws = [w for w in ws_axis if w in r['sweep']]
                if not ws:
                    continue
                wb = min(ws, key=lambda w: r['sweep'][w]['fid'])
                yb = r['sweep'][wb]['fid']
                ax.plot([pos[wb]], [yb], marker='o', ms=13, mfc='none',
                        mec=r['colour'], mew=2.0, zorder=6)
                ax.annotate(f'{yb:.3f}', (pos[wb], yb), textcoords='offset points',
                            xytext=(11, -4), ha='left', fontsize=9,
                            fontweight='bold', color=r['colour'], zorder=7)
            ax.margins(x=0.12)
        if is_fid:
            # baseline reference line only -- its value is already on its own ring
            best = min(runs[-1]['sweep'][w]['fid'] for w in runs[-1]['sweep'])
            ax.axhline(best, color=BASE, ls=':', lw=1.2, zorder=1)
        ax.set_title(f'{title}   (CFG region $w$ = {W_MIN:g}-{W_MAX:g})',
                     loc='left', fontsize=12, pad=10)
        ax.set_xticks(range(len(ws_axis)))
        ax.set_xticklabels([f'{w:g}' for w in ws_axis])
        ax.set_xlabel('CFG guidance scale  $w$')
        ax.grid(alpha=0.25, lw=0.7)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[0, 1].legend(fontsize=9, loc='lower right', frameon=False,
                      title='arm', title_fontsize=9)
    fig.suptitle('reg_ft + per-step MMD: guidance sweep  (4 MMD arms vs baseline)',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.935,
             'All four MMD arms beat the reg_ft baseline slightly and by the same '
             'margin: best FID 4.29-4.34 against 4.374, optimum near w=2.2-2.4.\n'
             'A 100x change in mmd_weight (10 -> 1000) moves every metric less than '
             'that margin, and moving the band down and bounding it (window, dotted) '
             'lands in the same place.',
             ha='center', va='top', fontsize=10, color='#555555')
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    fig.savefig('plots/wsweep_mmd.png', dpi=150)
    print('wrote plots/wsweep_mmd.png')


if __name__ == '__main__':
    main()
