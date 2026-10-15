"""Audit complete c16 mechanisms, report paired effects and critic diagnostics."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_stats',ROOT/'src/rf_backdoor/magmaw_effective.py')
stats=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    data=json.loads(args.metrics.read_text())
    identity=data['identity']
    if data['status']!='completed' or not data['codecs_unchanged'] or data['codec_parameter_gradients']:
        raise ValueError('incomplete/codec changed or accumulated gradients')
    if identity['P0_sha256']!='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542':
        raise ValueError('wrong original P0')
    if identity['generator_inputs']!=['z ~ Uniform(0,1), 100 dimensions'] or not identity['white_box_P0_training']:
        raise ValueError('unexpected generator input/threat model')
    if identity['arguments']['smoke']: raise ValueError('smoke results cannot be formal results')
    summaries=[]
    repeats=identity['arguments']['evaluation_repeats']
    profiles=set(identity['arguments']['profiles'])|{'warm_shared_reference'}
    if set(data['profiles'])!=profiles: raise ValueError('missing profile')
    for name,profile in data['profiles'].items():
        if name!='warm_shared_reference':
            if profile['diversity_probe']['normalized_mean_distance']<.001:
                raise ValueError('random-code collapse: modules are present but diversity is ineffective')
            for record in profile['history']:
                if any(not math.isfinite(v) for v in record.values() if isinstance(v,(float,int))):
                    raise ValueError('invalid training metric')
            if len(profile['history'])!=identity['arguments']['steps']: raise ValueError('training incomplete')
            if name=='balanced' and not any(r['beta_ds']>0 and r['beta_dv']>0 for r in profile['history']):
                raise ValueError('requested regularizers never activated')
            if name=='regularizer_off' and any(r['beta_ds'] or r['beta_dv'] for r in profile['history']):
                raise ValueError('invalid ablation')
        groups={}
        for row in profile['rows']:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
                raise ValueError('nonfinite result')
            if max(row['received_psr_db'],row['constraint_psr_db'])>row['budget_db']+1e-4:
                raise ValueError('power exceeded')
            if abs(row['received_psr_db']-row['random_received_psr_db'])>1e-4:
                raise ValueError('power control mismatch')
            groups.setdefault((row['snr_db'],row['budget_db']),[]).append(row)
        if set(groups)!={(s,b) for s in [10,25,100] for b in [-20,-18,-16,-14,-12,-10]}:
            raise ValueError('missing conditions')
        for (snr,budget),rows in sorted(groups.items()):
            if len(rows)!=50*repeats or {(r['key'],r['repeat']) for r in rows}!={(k,i) for k in range(50,100) for i in range(repeats)}:
                raise ValueError('sample identity mismatch')
            item={'profile':name,'snr_db':snr,'budget_db':budget,'records':len(rows)}
            item.update(stats.paired_summary(*[[r[k] for r in rows] for k in ['clean','attack','random','key']]))
            for key in ['clean','attack','random','received_psr_db','pre_cap_received_psr_db']:
                item[key]=statistics.mean(r[key] for r in rows)
            if name!='warm_shared_reference':
                item['critic_balanced_accuracy']=(statistics.mean(r['critic_clean_correct'] for r in rows)+
                                                 statistics.mean(r['critic_attack_correct'] for r in rows))/2
            summaries.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        fields=sorted(set().union(*(r.keys() for r in summaries)))
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,snr in zip(axes,[10,100]):
            for name in data['profiles']:
                rows=sorted([r for r in summaries if r['profile']==name and r['snr_db']==snr],key=lambda r:r['budget_db'])
                ax.plot([r['budget_db'] for r in rows],[r['attack'] for r in rows],'o-',label=name)
            ax.axhline(rows[0]['clean'],color='black',linestyle='--',label='Frozen original P0 clean')
            ax.set(title=f'Full EOT test, SNR {snr} dB',xlabel='Received PSR cap (dB)',ylabel='Reconstruction PSNR (dB)')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(args.output_dir/'complete_c16_psnr.png',dpi=160)
        plt.close(fig)
    print(json.dumps({'audit':'passed','records':sum(r['records'] for r in summaries),
        'SNR10_minus10':[r for r in summaries if r['snr_db']==10 and r['budget_db']==-10],
        'diversity':{k:v.get('diversity_probe') for k,v in data['profiles'].items()}},indent=2))


if __name__=='__main__': main()
