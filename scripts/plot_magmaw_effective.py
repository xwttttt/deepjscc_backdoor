"""Plot held-out paired-power results, averaged over all six protocols."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for run in ['primary','confirm']:
        overview=json.loads((args.directory/run/'report/overview.json').read_text())
        seed=1234 if run=='primary' else 5678
        for name,profile in overview.items():
            data=profile['by_budget']
            budgets=sorted(map(int,data))
            label=f'{name}, seed {seed}'
            values=[data[str(x)]['drop_db'] for x in budgets]
            intervals=np.array([data[str(x)]['drop_ci95'] for x in budgets]).T
            axes[0].errorbar(budgets,values,yerr=[np.maximum(0,np.array(values)-intervals[0]),np.maximum(0,intervals[1]-np.array(values))],marker='o',capsize=3,label=label)
            axes[0].plot(budgets,[data[str(x)]['random_drop_db'] for x in budgets],'--',alpha=.55,label=f'paired random ({name}, {seed})')
            values=[data[str(x)]['advantage_db'] for x in budgets]
            intervals=np.array([data[str(x)]['advantage_ci95'] for x in budgets]).T
            axes[1].errorbar(budgets,values,yerr=[np.maximum(0,np.array(values)-intervals[0]),np.maximum(0,intervals[1]-np.array(values))],marker='o',capsize=3,label=label)
    axes[0].set(title='Held-out PSNR drop: six-protocol mean',xlabel='Receive PSR cap (dB)',ylabel='Mean PSNR drop (dB)')
    axes[1].set(title='Advantage over same-power random',xlabel='Receive PSR cap (dB)',ylabel='Random PSNR minus learned PSNR (dB)')
    axes[1].axhline(0,color='black',ls='--',alpha=.4)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(args.directory/'effectiveness.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
