"""Plot original c16 quality, universal attacks and separately labeled adaptive reference."""
import json
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    backdoor=json.loads((args.directory/'backdoor_comparison.json').read_text())['summary']
    pgd=json.loads((args.directory/'primary/report/pgd_summary.json').read_text())
    for ax,snr in zip(axes,[10,100]):
        for run in ['primary','confirm']:
            summary=json.loads((args.directory/run/'report/summary.json').read_text())
            profiles=sorted({row['profile'] for row in summary})
            for profile in profiles:
                rows=sorted([row for row in summary if row['profile']==profile and row['snr_db']==snr],key=lambda r:r['budget_db'])
                ax.plot([r['budget_db'] for r in rows],[r['attack'] for r in rows],marker='o',label=f"UAP {profile}, seed {rows[0]['seed']}")
                if run=='primary' and profile==profiles[0]:
                    ax.plot([r['budget_db'] for r in rows],[r['random'] for r in rows],'--',alpha=.5,label='Random, matched UAP power')
                    ax.axhline(rows[0]['clean'],color='black',linestyle='--',label='Original frozen P0 clean')
        record=next(row for row in backdoor if row['snr_db']==snr)
        ax.axhline(record['backdoor_triggered'],color='red',linestyle=':',label='Saved backdoor trigger (own constraint)')
        if snr==10:
            ax.plot([r['budget_db'] for r in pgd],[r['attack'] for r in pgd],'kD-',label='Per-image + known Ha, 80-step reference')
        ax.set(title=f'Original c16 P0: SNR {snr} dB',xlabel='Received attack PSR cap (dB)',ylabel='Reconstruction PSNR (dB)')
        ax.legend(fontsize=6.5)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.directory/'c16_psnr.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
