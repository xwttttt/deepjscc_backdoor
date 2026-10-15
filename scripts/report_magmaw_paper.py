"""Audit constraints and aggregate equation-level Magmaw profiles."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.metrics.read_text(encoding='utf-8'))
    if report['status']!='completed':
        raise ValueError('incomplete run')
    repeats=report['identity']['arguments']['evaluation_repeats']
    steps=report['identity']['arguments']['steps']
    summaries=[]
    for name,profile in report['profiles'].items():
        grouped={}
        for row in profile['test']:
            grouped.setdefault((row['c'],row['modulation'],row['psr_budget_db']),[]).append(row)
        if len(grouped)!=36:
            raise ValueError('missing protocol/budget condition')
        for (c,degree,budget),rows in sorted(grouped.items()):
            identities={(r['key'],r['repeat']) for r in rows}
            if len(rows)!=50*repeats or identities!={(key,repeat) for key in range(50,100) for repeat in range(repeats)}:
                raise ValueError('missing/duplicate test sample identity')
            # Before-channel profile constrains frequency energy. Legacy profile
            # projects its entire long received burst; a shorter capture window
            # is not itself constrained. That full-interval energy was not saved
            # in the rows, so do not claim to audit it from capture-window power.
            if profile['projection']=='before_channel' and any(r['constraint_psr_db']>budget+1e-4 for r in rows):
                raise ValueError('declared projection constraint exceeded')
            item={'profile':name,'steps':steps,'c':c,'modulation':degree,'budget_db':budget,
                  'records':len(rows),'selected_step':profile['best_step'],
                  'constraint_audit':'frequency_pass' if profile['projection']=='before_channel' else 'full_interval_not_in_rows'}
            for key in ['clean','attack','drop_db','constraint_psr_db','launch_psr_db','received_psr_db']:
                item[key]=statistics.mean(r[key] for r in rows)
            summaries.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (args.output_dir/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({name:{'max_test_drop_db':max(r['drop_db'] for r in summaries if r['profile']==name),
                           'validation_drop_db':v['validation_drop_db'],'best_step':v['best_step']}
                      for name,v in report['profiles'].items()},indent=2))


if __name__=='__main__':
    main()
