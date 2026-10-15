"""Audit validation-only strength selection and paired test improvement."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired',ROOT/'src/rf_backdoor/magmaw_effective.py')
paired=importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    data=json.loads(args.metrics.read_text())
    if data['status']!='completed' or not data['codecs_unchanged'] or data['codec_gradients']:
        raise ValueError('codec freeze failed/incomplete')
    identity=data['identity']
    if identity['arguments']['smoke']: raise ValueError('smoke is not a formal result')
    if data['test_evaluation_started_at_unix']<data['training_complete']['completed_at_unix']:
        raise ValueError('test was used before model selection finished')
    if identity['P0_sha256']!='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542':
        raise ValueError('wrong P0')
    if data['training_complete']['source_sha256']!=identity['code_sha256']:
        raise ValueError('training/evaluation source identity mismatch')
    selection=json.loads((args.metrics.parent/'pilot_selection.json').read_text())
    chosen=max(selection['candidates'],key=lambda n:selection['candidates'][n]['validation_score'])
    if chosen!=selection['selected_profile'] or chosen!=data['training_complete']['selected_profile']:
        raise ValueError('pilot selection did not use the saved validation maximum')
    refinements=data['training_complete']['refinements']
    if max(refinements,key=lambda n:refinements[n]['validation_score'])!=data['training_complete']['selected_refinement']:
        raise ValueError('seed selection did not use the saved validation maximum')
    gradient_probes=0
    for record in [*selection['candidates'].values(),*refinements.values()]:
        history=json.loads((args.metrics.parent/record['name']/'history.json').read_text())
        if len(history)!=record['steps_executed'] or [r['step'] for r in history]!=list(range(1,len(history)+1)):
            raise ValueError('incomplete training history')
        for validation in record.get('validations',[]):
            if abs(validation['score']-(.7*validation['focused_drop_db']+.3*validation['mean_drop_db']))>1e-9:
                raise ValueError('validation score definition changed')
        for row in history:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
                raise ValueError('invalid training scalar')
            for term,probe in row['gradient_diagnostics'].items():
                gradient_probes+=1
                if any(not math.isfinite(v) for v in probe.values()): raise ValueError('invalid gradient probe')
                if probe['projected_dot'] < -1e-5*max(1e-12,probe['primary_l2']**2):
                    raise ValueError('opposing gradient survived projection')
                if probe['applied_aux_l2']>identity['auxiliary_gradient_caps'][term]*probe['primary_l2']+1e-5:
                    raise ValueError('auxiliary gradient exceeded cap')
    expected={f'refine_{seed}' for seed in identity['arguments']['seeds']}|{'previous_regularizer_off','previous_balanced','shared_reference'}
    if set(data['profiles'])!=expected: raise ValueError('missing candidates/references')
    repeats=identity['arguments']['evaluation_repeats']
    groups={}
    for name,result in data['profiles'].items():
        if name.startswith('refine_') and result['diversity_distance']<.001: raise ValueError('latent collapse')
        for row in result['rows']:
            if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))): raise ValueError('nonfinite result')
            if max(row['received_psr_db'],row['constraint_psr_db'])>row['budget_db']+1e-4: raise ValueError('power exceeded')
            if abs(row['random_received_psr_db']-row['received_psr_db'])>1e-4: raise ValueError('random power mismatched')
            if not 0<=row['fraction_blocks_drop_ge_3db']<=1: raise ValueError('invalid coverage')
            groups.setdefault((name,row['snr_db'],row['budget_db']),[]).append(row)
    expected_groups={(name,s,b) for name in expected for s in [10,25,100] for b in [-20,-18,-16,-14,-12,-10]}
    if set(groups)!=expected_groups: raise ValueError('missing conditions')
    summaries=[]
    for (name,snr,budget),rows in sorted(groups.items()):
        if len(rows)!=50*repeats or {(r['key'],r['repeat']) for r in rows}!={(k,i) for k in range(50,100) for i in range(repeats)}:
            raise ValueError('sample identity mismatch')
        reference={(r['key'],r['repeat']):r for r in groups[('previous_regularizer_off',snr,budget)]}
        old=[reference[(r['key'],r['repeat'])] for r in rows]
        if any(abs(r['clean']-a['clean'])>2e-4 for r,a in zip(rows,old)): raise ValueError('unpaired clean baseline')
        item={'profile':name,'snr_db':snr,'budget_db':budget,'records':len(rows)}
        item.update(paired.paired_summary(*[[r[key] for r in rows] for key in ['clean','attack','random','key']]))
        improvement=paired.paired_summary([r['attack'] for r in old],[r['attack'] for r in rows],
                                          [r['attack'] for r in old],[r['key'] for r in rows])
        item['improvement_vs_previous_off_db']=improvement['drop_db']
        item['improvement_ci95']=improvement['drop_ci95']
        for key in ['clean','attack','random','received_psr_db','fraction_blocks_drop_ge_3db']:
            item[key]=statistics.mean(r[key] for r in rows)
        summaries.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.6))
        for ax,snr in zip(axes,[10,100]):
            for name in sorted(expected):
                rows=sorted([r for r in summaries if r['profile']==name and r['snr_db']==snr],key=lambda r:r['budget_db'])
                ax.plot([r['budget_db'] for r in rows],[r['attack'] for r in rows],'o-',label=name)
            ax.axhline(rows[0]['clean'],color='black',linestyle='--',label='Original P0 clean')
            ax.set(title=f'Full physical transforms, SNR {snr} dB',xlabel='PSR cap (dB)',ylabel='Reconstruction PSNR (dB)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir/'priority_psnr.png',dpi=160)
        plt.close(fig)
    print(json.dumps({'audit':'passed','records':sum(r['records'] for r in summaries),'gradient_probes':gradient_probes,
        'selected_profile':data['training_complete']['selected_profile'],
        'selected_refinement':data['training_complete']['selected_refinement'],
        'snr10_minus10':[r for r in summaries if r['snr_db']==10 and r['budget_db']==-10]},indent=2))


if __name__=='__main__': main()
