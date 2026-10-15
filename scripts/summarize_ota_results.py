"""Summarize paired, single-seed test results; generate CSV/JSON and a plot."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    files = sorted(args.directory.glob('psr_*_metrics.json'))
    summaries = []
    for file in files:
        metrics = json.loads(file.read_text(encoding='utf-8'))
        rows = metrics['test']['samples']
        item = {'psr_db': metrics['config']['psr_db'], 'count': len(rows), **metrics['test']['mean']}
        for name, left, right in [('uap_drop', 'clean', 'uap'), ('extra_drop', 'random', 'uap')]:
            values = [r[left] - r[right] for r in rows]
            mean = statistics.mean(values)
            # t_49,0.975 for this fixed 50-sample experiment.
            if len(values) != 50:
                raise ValueError('this report requires exactly 50 held-out samples')
            half = 2.009575 * statistics.stdev(values) / math.sqrt(len(values))
            item[name + '_sample_ci95_low'] = mean - half
            item[name + '_sample_ci95_high'] = mean + half
        item['received_psr_max_error_db'] = max(abs(r['received_psr_db'] - item['psr_db']) for r in rows)
        summaries.append(item)
    summaries.sort(key=lambda r: r['psr_db'])
    (args.directory/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    with (args.directory/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, key in [('Clean', 'clean'), ('Matched-power random', 'random'), ('Learned UAP', 'uap')]:
        ax.plot([s['psr_db'] for s in summaries], [s[key] for s in summaries], marker='o', label=label)
    ax.set(xlabel='Received interference-to-signal power ratio (dB)', ylabel='Mean image PSNR (dB)',
           title='DeepJSCC: ideal-equalized OTA channel simulation')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.directory/'psnr_vs_psr.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
