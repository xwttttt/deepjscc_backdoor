"""Audit preserved c16 identity, paired power, fixed test keys and SNR matrix."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('effective_summary',ROOT/'src/rf_backdoor/magmaw_effective.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.metrics.read_text())
    if report['status']!='completed' or not report['p0_weights_unchanged']:
        raise ValueError('incomplete or codec changed')
    if report['identity']['c']!=16 or report['identity']['checkpoint_sha256']!='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542':
        raise ValueError('c16 P0 identity mismatch')
    for gate in report['baseline_checks'].values():
        if gate['max_paired_psnr_difference_db']>2e-4:
            raise ValueError('OFDM altered the paired original P0 baseline')
    repeats=report['identity']['arguments']['evaluation_repeats']
    summaries=[]
    for name,profile in report['profiles'].items():
        groups={}
        for row in profile['test']:
            if max(row['received_psr_db'],row['constraint_psr_db'])>row['psr_budget_db']+1e-4:
                raise ValueError('budget exceeded')
            if abs(row['received_psr_db']-row['random_received_psr_db'])>1e-4:
                raise ValueError('random control power mismatch')
            groups.setdefault((row['snr_db'],row['psr_budget_db']),[]).append(row)
        if set(groups)!={(snr,budget) for snr in [10,25,100] for budget in [-20,-18,-16,-14,-12,-10]}:
            raise ValueError('missing SNR/budget condition')
        for (snr,budget),rows in sorted(groups.items()):
            if len(rows)!=50*repeats or {(r['key'],r['repeat']) for r in rows}!={(k,i) for k in range(50,100) for i in range(repeats)}:
                raise ValueError('missing/duplicate test identity')
            item={'profile':name,'seed':report['identity']['arguments']['seed'],'snr_db':snr,
                  'budget_db':budget,'best_step':profile['best_step'],'records':len(rows)}
            item.update(module.paired_summary(*[[r[k] for r in rows] for k in ['clean','attack','random','key']]))
            for key in ['clean','attack','random','received_psr_db','pre_cap_received_psr_db']:
                item[key]=statistics.mean(r[key] for r in rows)
            summaries.append(item)
    pgd_summary=[]
    for budget in [-16,-10]:
        rows=[r for r in report['pgd_reference'] if r['psr_budget_db']==budget]
        if not rows:
            continue
        if len(rows)!=50 or {r['key'] for r in rows}!=set(range(50,100)) or any(r['received_psr_db']>budget+1e-4 for r in rows):
            raise ValueError('invalid PGD reference')
        item={'profile':'image_and_channel_conditioned_PGD_reference','budget_db':budget,'snr_db':10,'records':len(rows)}
        item.update(module.paired_summary(*[[r[k] for r in rows] for k in ['clean','attack','random','key']]))
        for key in ['clean','attack','random','received_psr_db']:
            item[key]=statistics.mean(r[key] for r in rows)
        pgd_summary.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
    (args.output_dir/'pgd_summary.json').write_text(json.dumps(pgd_summary,indent=2))
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({'audit':'passed','universal_records':sum(r['records'] for r in summaries),
                      'pgd_records':sum(r['records'] for r in pgd_summary),'snr10': [r for r in summaries if r['snr_db']==10],
                      'pgd':pgd_summary},indent=2))


if __name__=='__main__':
    main()
