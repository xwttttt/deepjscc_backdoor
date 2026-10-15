"""CPU audit, paired cluster statistics, and explicit ablation plots."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_stats', ROOT/'src/rf_backdoor/magmaw_effective.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    data = json.loads(args.metrics.read_text())
    if data['status'] != 'completed' or not data['p0_weights_unchanged'] or data['target_parameter_gradients']:
        raise ValueError('incomplete or invalid model isolation')
    if data['identity']['p0_sha256'] != '24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542':
        raise ValueError('P0 identity mismatch')
    repeats = data['identity']['arguments']['evaluation_repeats']
    summaries = []
    for name, record in data['results'].items():
        groups = {}
        for row in record['rows']:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                raise ValueError('nonfinite metric')
            if max(row['received_psr_db'], row['constraint_psr_db']) > row['budget_db'] + 1e-4:
                raise ValueError('exceeded power constraint')
            if abs(row['random_received_psr_db'] - row['received_psr_db']) > 1e-4:
                raise ValueError('unmatched random control')
            groups.setdefault((row['case'], row['budget_db']), []).append(row)
        if set(groups) != {(case, budget) for case in ['full', 'no_permutation', 'aligned_reference'] for budget in [-16, -10]}:
            raise ValueError('missing ablation conditions')
        for (case, budget), rows in sorted(groups.items()):
            if len(rows) != 50*repeats or {(r['key'], r['repeat']) for r in rows} != {(k, i) for k in range(50, 100) for i in range(repeats)}:
                raise ValueError('missing/duplicated test identity')
            item = {'variant': name, 'test_case': case, 'budget_db': budget, 'records': len(rows),
                    'train_case': record['train_case']}
            item.update(stats.paired_summary(*[[r[key] for r in rows] for key in ['clean','attack','random','key']]))
            for key in ['clean', 'attack', 'random', 'received_psr_db', 'pre_cap_received_psr_db']:
                item[key] = statistics.mean(r[key] for r in rows)
            summaries.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries, indent=2))
    with (args.output_dir/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        labels = {'shared32_full':'Shared 32, trained full EOT',
                  'shared1408_full':'Shared 1408, trained full EOT',
                  'shared1408_no_permutation':'Shared 1408, trained without permutation',
                  'shared1408_aligned_reference':'Shared 1408, trained aligned (diagnostic)',
                  'previous_pgm_psnr':'Previous random-z PGM'}
        for ax, case in zip(axes, ['full', 'no_permutation', 'aligned_reference']):
            for name in data['results']:
                rows = sorted([r for r in summaries if r['variant']==name and r['test_case']==case], key=lambda r:r['budget_db'])
                ax.plot([r['budget_db'] for r in rows], [r['attack'] for r in rows], 'o-', label=labels[name])
            ax.axhline(rows[0]['clean'], color='black', linestyle='--', label='Original P0 clean')
            ax.set(title=case.replace('_', ' ')+'; SNR 10 dB', xlabel='PSR cap (dB)', ylabel='Reconstruction PSNR (dB)')
            ax.set_xticks([-16, -10])
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.tight_layout(rect=(0,.17,1,1))
        fig.savefig(args.output_dir/'uap_diagnosis.png', dpi=160)
        plt.close(fig)
    print(json.dumps({'audit':'passed', 'records':sum(r['records'] for r in summaries),
                      'full_minus10':[r for r in summaries if r['test_case']=='full' and r['budget_db']==-10]}, indent=2))


if __name__ == '__main__':
    main()
