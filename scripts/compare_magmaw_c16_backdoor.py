"""Server evaluation: same c16 data/noise/link, saved backdoor weights/trigger."""
import json
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.impairments import IQDCOffsetConfig,apply_iq_dc_torch
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_c16 import build_analog_link
    if not torch.cuda.is_available():
        raise RuntimeError('server CUDA required')
    torch.set_num_threads(4)
    models={}
    checkpoints={'p0':('/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt',
                       '24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542'),
                 'backdoor':('/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260928/stronger_struct_60ep_001/best.pt',
                             'db7fcc8e4bacb743ae780ffdb1a0516319b82d96f5a90893e5ac64182e5d166c')}
    for name,(path,expected) in checkpoints.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('checkpoint mismatch: '+name)
        model=DeepJSCCAdapter.from_local(c=16)
        model.model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model'],strict=True)
        model.model.to('cuda').eval().requires_grad_(False)
        models[name]=model
    trigger_json=json.loads((ROOT/'stronger_trigger.json').read_text())
    trigger=IQDCOffsetConfig(**trigger_json['config'])
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    bank=channel_bank(2000,20261014+200,'cuda')
    rows=[]
    with torch.no_grad():
        for snr in [10,25,100]:
            link=build_analog_link('cuda',snr)
            for repeat in range(5):
                torch.manual_seed(20261014+1000+repeat)
                key=50
                for images,_ in DataLoader(dataset,batch_size=8):
                    images=images.to('cuda')
                    # Noise sampled once from P0 reference, shared across all
                    # branches exactly as the original paired drop objective.
                    p0_latent=models['p0'].encode(images)
                    state=link.prepare(p0_latent,bank=bank)
                    noise_latent=state['direct_rx']-p0_latent
                    # Replay the UAP evaluator's draws so the next batch uses
                    # the same noise/Ha as every UAP budget and profile.
                    torch.rand(len(images),100,device='cuda')
                    torch.randn(len(images),32,64,device='cuda',dtype=torch.complex64)
                    values={}
                    for name,adapter in models.items():
                        latent=adapter.encode(images)
                        for triggered in [False,True]:
                            source=apply_iq_dc_torch(latent,trigger) if triggered else latent
                            # The OFDM baseline gate establishes this analog
                            # mapping equals the waveform capture data bins.
                            received=state['direct_rx'] if name=='p0' and not triggered else source+noise_latent
                            decoded=adapter.decode(received)
                            score=-10*torch.log10((images-decoded).square().mean((1,2,3)).clamp_min(1e-12))
                            values[name+('_triggered' if triggered else '_clean')]=score
                    for i in range(len(images)):
                        rows.append({'key':key+i,'repeat':repeat,'snr_db':snr,**{n:float(v[i]) for n,v in values.items()}})
                    key+=len(images)
                if dataset._env is not None:
                    dataset._env.close()
                    dataset._env=None
            print('backdoor comparison SNR',snr,'completed',flush=True)
    summary=[]
    for snr in [10,25,100]:
        group=[row for row in rows if row['snr_db']==snr]
        if len(group)!=250 or {(r['key'],r['repeat']) for r in group}!={(k,i) for k in range(50,100) for i in range(5)}:
            raise ValueError('comparison matrix mismatch')
        values={name:sum(row[name] for row in group)/len(group) for name in ['p0_clean','p0_triggered','backdoor_clean','backdoor_triggered']}
        summary.append({'snr_db':snr,**values,'p0_trigger_drop_db':values['p0_clean']-values['p0_triggered'],
            'backdoor_trigger_drop_db':values['backdoor_clean']-values['backdoor_triggered'],
            'backdoor_clean_utility_drop_db':values['p0_clean']-values['backdoor_clean']})
    output={'status':'completed','scope':'same DIV2K keys, common AWGN and continuous c16 receiver; trigger after normalization',
            'checkpoint_sha256':{name:value[1] for name,value in checkpoints.items()},
            'trigger':trigger_json,'summary':summary,'rows':rows,
            'evaluation_seed':20261014,'uap_evaluation_draw_replay':True,
            'limitations':'trigger acts on normalized latent; waveform OTA has independent Ha. Common receiver/noise, not identical attack information/power constraints.'}
    (ROOT/'backdoor_comparison.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
