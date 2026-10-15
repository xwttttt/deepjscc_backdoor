"""Audit paired receive power, and report key-clustered PSNR confidence intervals."""
import argparse
import csv
import json
from pathlib import Path
import sys
import statistics
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('effective_summary',ROOT/'src/rf_backdoor/magmaw_effective.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
paired_summary=module.paired_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.metrics.read_text(encoding='utf-8'))
    if report['status']!='completed':
        raise ValueError('run incomplete')
    repeats=report['identity']['arguments']['evaluation_repeats']
    summary=[]
    overview={}
    for name,profile in report['profiles'].items():
        groups={}
        for row in profile['test']:
            if row['received_psr_db']>row['psr_budget_db']+1e-4:
                raise ValueError('receive budget exceeded')
            if row['constraint_psr_db']>row['psr_budget_db']+1e-4:
                raise ValueError('frequency budget exceeded')
            if abs(row['random_received_psr_db']-row['received_psr_db'])>1e-4:
                raise ValueError('random baseline power mismatch')
            groups.setdefault((row['c'],row['modulation'],row['psr_budget_db']),[]).append(row)
        if len(groups)!=36:
            raise ValueError('missing condition')
        for (c,modulation,budget),rows in sorted(groups.items()):
            if len(rows)!=50*repeats or {(r['key'],r['repeat']) for r in rows}!={(k,i) for k in range(50,100) for i in range(repeats)}:
                raise ValueError('sample identity mismatch')
            result={'profile':name,'seed':report['identity']['arguments']['seed'],
                    'steps':report['identity']['arguments']['steps'],'best_step':profile['best_step'],
                    'c':c,'modulation':modulation,'budget_db':budget,'records':len(rows)}
            result.update(paired_summary(*[[r[k] for r in rows] for k in ['clean','attack','random','key']]))
            for key in ['clean','attack','random','received_psr_db','pre_cap_received_psr_db']:
                result[key]=statistics.mean(r[key] for r in rows)
            summary.append(result)
        overview[name]={'selected_step':profile['best_step'],'proxy_validation_drop_db':profile['validation_drop_db'],
                        'records':len(profile['test']),'by_budget':{}}
        for budget in [-20,-18,-16,-14,-12,-10]:
            rows=[r for r in profile['test'] if r['psr_budget_db']==budget]
            overview[name]['by_budget'][str(budget)]=paired_summary(*[[r[k] for r in rows] for k in ['clean','attack','random','key']])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output_dir/'overview.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
    with (args.output_dir/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(overview,indent=2))


if __name__=='__main__':
    main()
