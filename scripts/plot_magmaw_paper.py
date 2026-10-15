"""Plot fixed-budget diagnosis and measured receive power separately."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    directory=args.directory
    summary=json.loads((directory/'diagnosis_summary.json').read_text(encoding='utf-8'))
    metrics=json.loads((directory/'diagnosis_metrics.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    names=['literal','mean_diversity','no_diversity','distortion_only','legacy_projection']
    for name in names:
        rows=[r for r in summary if r['profile']==name and r['c']==8 and r['modulation']==16]
        axes[0].plot([r['budget_db'] for r in rows],[r['drop_db'] for r in rows],marker='o',label=name)
        axes[1].plot([r['budget_db'] for r in rows],[r['received_psr_db'] for r in rows],marker='o',label=name)
    axes[0].set(title='400-step diagnosis: rate 1/6, 16-QAM',xlabel='Projection PSR budget (dB)',ylabel='Mean target PSNR drop (dB)')
    axes[1].plot([-20,-10],[-20,-10],'k--',alpha=.5)
    axes[1].set(title='Actual received power is reported',xlabel='Projection PSR budget (dB)',ylabel='Mean captured receive PSR (dB)')
    labels=['Reconstruction','Critic','Diversity']
    for offset,name in enumerate(['literal','mean_diversity']):
        gradients=metrics['profiles'][name]['history'][0]['gradient_l2']
        values=[gradients[k] for k in ['reconstruction','discriminative','diversity']]
        axes[2].bar(np.arange(3)+offset*.35,values,width=.35,label=name)
    axes[2].set(title='Initial weighted gradient L2',yscale='log',ylabel='Gradient norm (log scale)')
    axes[2].set_xticks(np.arange(3)+.175,labels)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory/'diagnosis.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
