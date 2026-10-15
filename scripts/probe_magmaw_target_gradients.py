"""Server probe: target-vs-proxy/regularizer gradients on a fixed training batch."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from torch.nn import functional as F
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_complete import build_full_generator,generator_terms
    from rf_backdoor.magmaw_full import SignalDiscriminator,channel_bank
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_c16_surrogate import load_proxies
    from rf_backdoor.magmaw_paper import paper_attack
    from rf_backdoor.magmaw_effective import cap_received
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    if not torch.cuda.is_available(): raise RuntimeError('server CUDA required')
    torch.set_num_threads(4)
    previous=Path('/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_complete_c16_20261014_02')
    report=json.loads((previous/'runs/full/metrics.json').read_text())
    p0=DeepJSCCAdapter.from_local(c=16)
    p0.model.load_state_dict(torch.load(report['identity']['arguments']['checkpoint'],map_location='cpu',weights_only=True)['model'])
    p0.model.to('cuda').eval().requires_grad_(False)
    directory=Path(report['identity']['arguments']['proxy_dir'])
    records=json.loads((directory.parent/'proxy_metrics.json').read_text())['proxies']
    proxies=load_proxies(directory,records,'cuda')
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb')
    images=torch.stack([dataset[i][0] for i in range(8)]).to('cuda')
    dataset._env.close()
    bank=channel_bank(2000,2026102301,'cuda')
    rows=[]
    for profile in ['balanced','regularizer_off']:
        saved=torch.load(previous/'runs/full'/(profile+'_best.pt'),map_location='cuda',weights_only=True)
        generator=build_full_generator(saved['generator']['mean_iq']).eval()
        generator.load_state_dict(saved['generator'])
        critic=SignalDiscriminator().to('cuda').eval().requires_grad_(False)
        critic.load_state_dict(saved['critic'])
        for physical_seed in [2026102302,2026102303,2026102304]:
            torch.manual_seed(2026102300)
            a=generator(torch.rand(8,100,device='cuda'))
            b=generator(torch.rand(8,100,device='cuda'))
            link=build_analog_link('cuda',10)
            losses={}
            clean_logits=[]
            attack_logits=[]
            for index in range(3):
                with torch.no_grad():
                    latent,shape=(p0.encode(images),None) if index==0 else proxies[index-1].encode(images)
                    torch.manual_seed(physical_seed)
                    state=link.prepare(latent,bank=bank)
                delta,_=paper_attack(link,a,state,-10,max_symbols=1408)
                attacked=link.decode(state['rx']+cap_received(delta,state,-10),state)
                output=p0.decode(attacked) if index==0 else proxies[index-1].decode(attacked,shape)
                mse=(images-output).square().mean((1,2,3)).clamp_min(1e-12)
                losses[f'model{index}_negative_mse']=-mse.mean()
                if index==0:
                    losses['target_psnr']=(-10*torch.log10(mse)).mean()
                    local=F.avg_pool2d((images-output).square().mean(1,keepdim=True),32)
                    losses['spatial_psnr']=(-10*torch.log10(local.clamp_min(1e-12))).mean()
                clean_logits.append(critic(link.decode(state['rx'],state)))
                attack_logits.append(critic(attacked))
            terms=generator_terms([torch.tensor(0.,device='cuda')],clean_logits,attack_logits,a,b,'rms_normalized_mean')
            losses['discriminator']=terms['ds']
            losses['negative_diversity']=-terms['dv']
            gradients={name:torch.autograd.grad(loss,generator.mean_iq,retain_graph=True)[0].flatten() for name,loss in losses.items()}
            primary=gradients['target_psnr']
            rows.append({'profile':profile,'physical_seed':physical_seed,
                'primary_gradient_l2':float(primary.norm()),'terms':{name:{'gradient_l2':float(g.norm()),
                    'cosine_with_target_psnr':float((primary@g)/(primary.norm()*g.norm()).clamp_min(1e-30))}
                    for name,g in gradients.items()}})
    (ROOT/'target_gradient_probe.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
