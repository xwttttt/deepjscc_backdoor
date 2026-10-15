"""Aggregate the full transfer matrix without treating channel repeats as images."""
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
        raise ValueError('only completed full reports may be aggregated')
    rows=report['test']
    methods=['random','vanilla_uap','sync_free_uap','magmaw']
    grouped={}
    for row in rows:
        grouped.setdefault((row['c'],row['modulation'],row['psr_db']),[]).append(row)
    cfg=report['identity']['config']
    expected=len(cfg['rates_c'])*len(cfg['modulations'])*len(cfg['psr_db'])
    if len(grouped)!=expected:
        raise ValueError(f'incomplete condition matrix: {len(grouped)}/{expected}')
    for (c,degree,budget),values in grouped.items():
        if len(values)!=50*cfg['evaluation_repeats']:
            raise ValueError('incomplete held-out sample/channel matrix')
        pairs={(r['key'],r['repeat']) for r in values}
        if pairs!={(key,repeat) for key in range(50,100) for repeat in range(cfg['evaluation_repeats'])}:
            raise ValueError('duplicate or missing sample/repeat identity')
        if any(v>budget+1e-4 for r in values for v in r['actual_psr_db'].values()):
            raise ValueError('received PSR exceeded its budget')
    summaries=[]
    for (c,degree,psr),values in sorted(grouped.items()):
        keys=sorted(set(r['key'] for r in values))
        item={'c':c,'coding_rate':c/48,'modulation':degree,'psr_db':psr,
              'unique_test_tiles':len(keys),'channel_realizations':len(values),
              'clean_psnr_db':statistics.mean(r['clean'] for r in values)}
        for method in methods:
            item[method+'_psnr_db']=statistics.mean(r[method] for r in values)
            item[method+'_drop_db']=statistics.mean(r['clean']-r[method] for r in values)
            item[method+'_max_received_psr_db']=max(r['actual_psr_db'][method] for r in values)
        item['magmaw_extra_vs_random_db']=item['random_psnr_db']-item['magmaw_psnr_db']
        item['magmaw_extra_vs_vanilla_db']=item['vanilla_uap_psnr_db']-item['magmaw_psnr_db']
        item['magmaw_extra_vs_syncfree_db']=item['sync_free_uap_psnr_db']-item['magmaw_psnr_db']
        summaries.append(item)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (args.output_dir/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for i,c in enumerate(sorted(set(r['c'] for r in summaries))):
        for j,degree in enumerate(sorted(set(r['modulation'] for r in summaries))):
            ax=axes[i,j]
            condition=[r for r in summaries if r['c']==c and r['modulation']==degree]
            ax.plot([r['psr_db'] for r in condition],[r['clean_psnr_db'] for r in condition],label='Clean',ls='--')
            for method in methods:
                ax.plot([r['psr_db'] for r in condition],[r[method+'_psnr_db'] for r in condition],marker='o',label=method)
            ax.set_title(f'Rate 1/{48//c}; '+('QPSK' if degree==4 else f'{degree}-QAM'))
            ax.grid(alpha=.2)
            if i==1: ax.set_xlabel('Received PSR budget (dB)')
            if j==0: ax.set_ylabel('Mean image PSNR (dB)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.suptitle('DIV2K DeepJSCC: held-out-model transfer, waveform OFDM simulation')
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(args.output_dir/'transfer_psnr_matrix.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'conditions':len(summaries),'records':len(rows),
                      'max_magmaw_drop_db':max(r['magmaw_drop_db'] for r in summaries)},indent=2))


if __name__=='__main__':
    main()
