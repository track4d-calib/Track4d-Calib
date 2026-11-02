#!/usr/bin/env python3
"""Plot measured calibration time against reconstruction quality."""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
import numpy as np


INK = '#243447'
MUTED = '#66788A'
BLUE = '#2376B7'
GRID = '#E5EBF0'


def style_axes(ax):
    ax.set_facecolor('#FCFDFE')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#BCC8D2')
    ax.tick_params(colors=MUTED, length=3, pad=6)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def draw_runtime_panel(ax, rows, baseline_name):
    """Show completed and failed attempts when their runtimes are close."""
    ax.clear()
    style_axes(ax)
    times = [row.get('calibration_seconds', row.get('attempt_seconds')) for row in rows]
    known = [value for value in times if value is not None]
    if any(not np.isfinite(value) or value <= 0 for value in known):
        raise ValueError('Attempt times must be finite and positive')
    maximum = max(known)
    labels = []
    for index, (row, seconds) in enumerate(zip(rows, times)):
        count = row.get('settings', {}).get('temporal_anchor_count')
        labels.append(f'{count} anchors' if count else textwrap.fill(row['label'], 14))
        if seconds is None:
            ax.text(index, maximum * .04, 'Failed\n(no timing)', ha='center', fontsize=8.5, color=MUTED)
            continue
        color = INK if row['name'] == baseline_name else BLUE if row['status'] == 'complete' else '#BCC8D2'
        ax.bar(index, seconds, width=.58, color=color, alpha=.85,
               hatch='//' if row['status'] != 'complete' else None, edgecolor='white')
        if 'registered_dynamic_poses' in row:
            coverage = f"{row['registered_dynamic_poses']}/{row['expected_dynamic_poses']} poses"
        elif row.get('missing_dynamic_poses'):
            coverage = f"{row['missing_dynamic_poses']} missing poses"
        else:
            coverage = 'Complete' if row['status'] == 'complete' else 'Unverified'
        ax.text(index, seconds + maximum * .025, f'{seconds:,.0f} s\n{coverage}',
                ha='center', va='bottom', fontsize=8.5, color=INK, linespacing=1.5)
    ax.set_xticks(range(len(rows)), labels, fontsize=8.5)
    ax.set_ylim(0, maximum * 1.30)
    ax.set_xlim(-.6, len(rows) - .4)
    ax.set_ylabel('Calibration runtime (seconds)', labelpad=9)
    ax.set_xlabel('Settings · hatched bars are incomplete runs', labelpad=10)
    ax.set_title('A   Runtime / pose coverage', loc='left', fontsize=11, weight='bold', pad=16)


def draw(summary, output):
    rows = summary['rows']
    if any(r['status'] not in ('complete', 'failed') for r in rows):
        raise ValueError('Wait for all planned cases before exporting the curve')
    valid = [r for r in rows if r['status'] == 'complete' and 'psnr' in r]
    baseline = next((r for r in rows if r['name'] == summary['baseline']
                     and r.get('calibration_seconds')), None)
    ours = [r for r in valid if r['name'] != summary['baseline']]
    if not baseline or len(ours) < 2:
        raise ValueError('Need baseline timing and at least two evaluated settings')
    if not np.isfinite(baseline['calibration_seconds']) or baseline['calibration_seconds'] <= 0:
        raise ValueError('Baseline time must be finite and positive')
    for row in valid:
        if not np.isfinite([row['calibration_seconds'], row['psnr']]).all() or row['calibration_seconds'] <= 0:
            raise ValueError('Times and metrics must be finite; times must be positive')
        expected = baseline['calibration_seconds'] / row['calibration_seconds']
        if not np.isclose(row['speedup'], expected):
            raise ValueError('Speedup does not match the recorded calibration times')
    ours.sort(key=lambda r: r['settings'].get('temporal_anchor_count', r['calibration_seconds']))
    xs = np.array([r['calibration_seconds'] for r in ours])
    ys = np.array([r['psnr'] for r in ours])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, (overview, detail) = plt.subplots(1, 2, figsize=(11.8, 5.5),
                                             gridspec_kw={'width_ratios': [1, 1.3]})
        fig.subplots_adjust(left=.075, right=.975, bottom=.29, top=.76, wspace=.29)
        fig.text(.075, .94, summary['protocol'].get('scene', 'Calibration benchmark'),
                 fontsize=20, weight='bold')
        fig.text(.075, .88, 'Calibration speed / reconstruction quality', fontsize=12, color=MUTED)
        for ax in (overview, detail):
            style_axes(ax)
            ax.plot(xs, ys, color=BLUE, linewidth=1.8, alpha=.8, zorder=2)
            ax.scatter(xs, ys, s=65, color=BLUE, edgecolors='white', linewidths=1.5, zorder=3)
            ax.set_xlabel('Complete calibration time (seconds)', labelpad=10)
            ax.set_ylabel('Reconstruction PSNR (dB)', labelpad=9)
        overview.set_title('A   Full comparison', loc='left', fontsize=11, weight='bold', pad=16)
        detail.set_title('B   Track4D-Calib settings · zoom', loc='left', fontsize=11, weight='bold', pad=16)
        overview.set_xscale('log')
        overview.set_xlim(min(min(xs), baseline['calibration_seconds']) / 1.55,
                          max(max(xs), baseline['calibration_seconds']) * 1.45)
        overview.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        overview.xaxis.set_major_formatter(ScalarFormatter())
        overview.minorticks_off()
        overview.set_xlabel('Calibration time (seconds, log scale)', labelpad=10)
        if 'psnr' in baseline:
            overview.scatter([baseline['calibration_seconds']], [baseline['psnr']], marker='D', s=65,
                             color=INK, edgecolors='white', linewidths=1.2, zorder=4)
            overview.annotate(f"{baseline['label']}\n{baseline['calibration_seconds']:,.0f} s · {baseline['psnr']:.2f} dB",
                              xy=(baseline['calibration_seconds'], baseline['psnr']),
                              xytext=(-12, 17), textcoords='offset points', ha='right', fontsize=9,
                              color=INK, linespacing=1.6)
        else:
            overview.axvline(baseline['calibration_seconds'], color=INK, linestyle=':', linewidth=1.3)
            overview.text(.03, .96, f"{baseline['label']}: {baseline['calibration_seconds']:,.0f} s\n"
                          f"PSNR unavailable · {baseline.get('missing_dynamic_poses', '?')} missing poses",
                          transform=overview.transAxes, va='top', fontsize=9, color=INK)
        overview.annotate('Track4D-Calib', xy=(xs[np.argmax(ys)], max(ys)),
                          xytext=(0, 17), textcoords='offset points', fontsize=10,
                          color=BLUE, weight='bold', ha='center')
        all_y = np.append(ys, baseline['psnr']) if 'psnr' in baseline else ys
        target = summary['protocol'].get('target_psnr')
        if target is not None:
            if not np.isfinite(target):
                raise ValueError('Target PSNR must be finite')
            overview.axhline(target, color=MUTED, linestyle='--', linewidth=1)
            overview.text(.03, target, f'{target:g} dB target',
                          transform=overview.get_yaxis_transform(), ha='left', va='bottom',
                          fontsize=9, color=MUTED)
            all_y = np.append(all_y, target)
        span = max(float(np.ptp(all_y)), .5)
        overview.set_ylim(min(all_y) - span * .25, max(all_y) + span * .5)
        xspan = max(float(np.ptp(xs)), 60)
        yspan = max(float(np.ptp(ys)), .25)
        detail.set_xlim(min(xs) - .22 * xspan, max(xs) + .22 * xspan)
        detail.set_ylim(min(ys) - .35 * yspan, max(ys) + .48 * yspan)
        detail.xaxis.set_major_locator(MaxNLocator(5))
        for index, row in enumerate(ours):
            offset = 16 if index % 2 == 0 else -33
            count = row['settings'].get('temporal_anchor_count')
            label = f'{count} anchor frames' if count is not None else row['label']
            detail.annotate(f"{label}\n{row['speedup']:.1f}× · {row['psnr']:.2f} dB",
                            xy=(row['calibration_seconds'], row['psnr']),
                            xytext=(0, offset), textcoords='offset points', ha='center',
                            fontsize=9, color=BLUE, linespacing=1.5,
                            bbox={'facecolor': '#FCFDFE', 'edgecolor': 'none', 'alpha': .9, 'pad': 1.5})
        all_times = np.append(xs, baseline['calibration_seconds'])
        if target is None and max(all_times) / min(all_times) < 3:
            draw_runtime_panel(overview, rows, summary['baseline'])
        protocol = summary['protocol']
        width, height = protocol['resolution']
        frame_ids = protocol['frame_ids']
        fig.text(.075, .125,
                 f"{protocol['iters']:,} training steps   ·   {width} × {height}   ·   "
                 f"{len(protocol['train_views'])} train / {len(protocol['test_views'])} test views   ·   "
                 f"{len(frame_ids)} frames   ·   seed {protocol['seed']}", fontsize=9, color=INK)
        fig.text(.075, .078, 'Shared images, preprocessing and point initialization; no checkpoint initialization. '
                 'Calibration includes all anchors.', fontsize=8.5, color=MUTED)
        failures = [r['label'] for r in rows if r['status'] != 'complete']
        note = 'Exploratory setting sweep · one timing per setting · straight segments connect measured settings'
        if failures:
            note += ' · No quality score: ' + ', '.join(failures)
        fig.text(.075, .035, note, fontsize=8, color=MUTED)
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output.with_suffix('.' + extension), dpi=220, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output filename stem')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    draw(summary, args.output)
    metadata = {'summary': str(args.summary),
                'summary_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'line_interpretation': 'Measured settings connected in anchor-count order; no fitted model',
                'style': {'font': 'DejaVu Sans', 'method': BLUE, 'baseline': INK}}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
