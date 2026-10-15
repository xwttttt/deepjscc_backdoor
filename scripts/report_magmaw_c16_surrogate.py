"""Audit proxy-only generation and report surrogate-to-original-P0 transfer."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_stats',ROOT/'src/rf_backdoor/magmaw_effective.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.metrics.read_text())
    attack=report['attack_complete']
    if report['status']!='completed' or not report['target_weights_unchanged'] or report['target_gradients']:
        raise ValueError('target freeze/gradient isolation failed')
    if attack['target_accessed'] or attack['target_queries']!=0 or attack['target_gradients'] or not attack['proxy_weights_unchanged']:
        raise ValueError('proxy-only generation violated')
    if report['target_loaded_at_unix']<=attack['completed_at_unix']:
        raise ValueError('target loaded too early')
    repeats=report['arguments']['evaluation_repeats']
    groups={}
    for row in report['rows']:
        if any(not math.isfinite(value) for value in row.values() if isinstance(value,(float,int))):
            raise ValueError('nonfinite metric')
        if max(row['received_psr_db'],row['constraint_psr_db'])>row['budget_db']+1e-4:
            raise ValueError('power exceeded')
        if abs(row['random_received_psr_db']-row['received_psr_db'])>1e-4:
            raise ValueError('random power mismatch')
        groups.setdefault((row['method'],row['snr_db'],row['budget_db']),[]).append(row)
    expected={('surrogate_universal',snr,budget) for snr in [10,25,100] for budget in [-20,-18,-16,-14,-12,-10]}
    expected|={('surrogate_image_conditioned',10,budget) for budget in [-16,-10]}
    if set(groups)!=expected:
        raise ValueError('transfer matrix incomplete')
    summaries=[]
    for (method,snr,budget),rows in sorted(groups.items()):
        count=repeats if method=='surrogate_universal' else 1
        if len(rows)!=50*count or {(r['key'],r['repeat']) for r in rows}!={(key,i) for key in range(50,100) for i in range(count)}:
            raise ValueError('test identity missing/duplicated')
        item={'method':method,'snr_db':snr,'budget_db':budget,'records':len(rows)}
        item.update(module.paired_summary(*[[r[key] for r in rows] for key in ['clean','attack','random','key']]))
        for key in ['clean','attack','random','received_psr_db','pre_cap_received_psr_db']:
            item[key]=statistics.mean(r[key] for r in rows)
        if method=='surrogate_image_conditioned':
            proxy=[r for r in attack['adaptive_proxy_test'] if r['budget_db']==budget]
            if len(proxy)!=50 or {r['key'] for r in proxy}!=set(range(50,100)):
                raise ValueError('proxy adaptive identity missing')
            item['proxy_drop_db']=statistics.mean(r['proxy_drop_db'] for r in proxy)
        summaries.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        # Adaptive rows include the additional proxy result; fill missing cells.
        fields=sorted(set().union(*(row.keys() for row in summaries)))
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({'audit':'passed','records':len(report['rows']),
                      'snr10':[r for r in summaries if r['snr_db']==10]},indent=2))


if __name__=='__main__':
    main()
