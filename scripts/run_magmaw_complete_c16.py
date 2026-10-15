"""Complete image-branch mechanisms on frozen c16: G/D, diversity, ensemble, EOT.

This remains white-box P0 training. Generator receives z only. Balanced mode
is an explicit full-length adaptation, not a claim of literal paper replication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
P0_SHA='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def save(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)


def close(data):
    if data._env is not None:
        data._env.close()
        data._env=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--proxy-dir',type=Path,required=True)
    p.add_argument('--warm-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--eot-samples',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--evaluation-repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--profiles',nargs='+',default=['balanced','regularizer_off'])
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.eot_samples,args.batch_size,args.evaluation_repeats)<1 or any(n not in ['balanced','regularizer_off','literal'] for n in args.profiles):
        p.error('positive sizes and known profiles required')
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank,SignalDiscriminator
    from rf_backdoor.magmaw_paper import paper_attack,diversity
    from rf_backdoor.magmaw_effective import cap_received,match_received_power
    from rf_backdoor.magmaw_c16_surrogate import load_proxies,state_fingerprint
    from rf_backdoor.magmaw_complete import build_full_generator,generator_terms,balanced_coefficients,permutation_probability
    if not torch.cuda.is_available(): raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    if sha(args.checkpoint)!=P0_SHA: raise ValueError('original c16 identity mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    p0=DeepJSCCAdapter.from_local(c=16)
    p0.model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    p0.model.to('cuda').eval().requires_grad_(False)
    records=json.loads((args.proxy_dir.parent/'proxy_metrics.json').read_text())['proxies']
    for r in records:
        if sha(args.proxy_dir/(r['name']+'.pt'))!=r['checkpoint_sha256']: raise ValueError('proxy identity mismatch')
    proxies=load_proxies(args.proxy_dir,records,'cuda')
    codecs=[p0.model,*proxies]
    fingerprints=[state_fingerprint(m) for m in codecs]
    # P0 is explicitly included in the ensemble: this is not a black-box experiment.
    weights=[1.,.25,.25]
    warm=torch.load(args.warm_checkpoint,map_location='cuda',weights_only=True)['raw_iq']
    if warm.shape!=(1,1408,64,2): raise ValueError('full-length shared warm start required')
    train,valid,test=load_div2k_splits(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    train_bank=channel_bank(2000,2026101701,'cuda')
    eval_bank=channel_bank(2000,2026101702,'cuda')
    identity={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'P0_sha256':P0_SHA,'proxy_sha256':{r['name']:r['checkpoint_sha256'] for r in records},
        'warm_checkpoint_sha256':sha(args.warm_checkpoint),'white_box_P0_training':True,
        'generator_inputs':['z ~ Uniform(0,1), 100 dimensions'],
        'ensemble_weights':weights,'frozen_models':['original_P0',*[r['name'] for r in records]],
        'fixed_attack_symbols':1408,'full_burst_basis_rank':8,
        'anti_collapse_design':'signed-z skip, bounded normalized gates, differentiable real-IQ QR, 5% stochastic RMS',
        'data_keys':{'train':[0,3200],'validation':[0,50],'test':[50,100]},
        'code_sha256':{name:sha(ROOT/name) for name in ['scripts/run_magmaw_complete_c16.py',
            'src/rf_backdoor/magmaw_complete.py','src/rf_backdoor/magmaw_c16_surrogate.py',
            'src/rf_backdoor/magmaw_c16.py','src/rf_backdoor/magmaw_paper.py','src/rf_backdoor/magmaw_full.py',
            'src/rf_backdoor/magmaw_effective.py','src/rf_backdoor/adapter.py','src/src/deepjscc/model.py','scripts/div2k_lmdb_dataset.py']},
        'limitations':['image-only/c16/continuous OFDM; no multimodal or modulation/rate sweep',
            'known unit legitimate AWGN; synthetic 8-tap Ha; no RF',
            'receiver-power clipping/matching remains an offline oracle',
            'P0 participates in training; ensemble is not a black-box transfer claim',
            'balanced diversity is RMS-normalized mean; literal raw L1 norm is separately audited',
            'gradient-balanced coefficients, curriculum and full-burst generator are local extensions']}
    save(args.output_dir/'identity.json',identity)

    def score(images,output):
        return -10*torch.log10((images-output).square().mean((1,2,3)).clamp_min(1e-12))

    def encode(index,images):
        return (p0.encode(images),None) if index==0 else proxies[index-1].encode(images)

    def decode(index,latent,shape):
        return p0.decode(latent) if index==0 else proxies[index-1].decode(latent,shape)

    def attack(link,pattern,state,budget):
        delta,powers=paper_attack(link,pattern,state,budget,max_symbols=1408)
        powers['pre_cap_received_psr_db']=powers['received_psr_db']
        delta=cap_received(delta,state,budget)
        powers['received_psr_db']=10*torch.log10(delta.abs().square().mean(-1)/state['power'][:,0])
        if not bool(torch.isfinite(delta).all()) or bool((powers['received_psr_db']>budget+1e-4).any()):
            raise AssertionError('attack invalid/over budget')
        return delta,powers

    def evaluate(pattern_fn,data,snrs,budgets,repeats,critic=None):
        rows=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            for snr in snrs:
                link=build_analog_link('cuda',snr)
                for budget in budgets:
                    for repeat in range(repeats):
                        key=data.start
                        for images,_ in DataLoader(data,batch_size=args.batch_size):
                            if args.smoke: images=images[:2]
                            images=images.to('cuda')
                            torch.manual_seed(2026101800+key+repeat*1000)
                            state=link.prepare(p0.encode(images),bank=eval_bank)
                            torch.manual_seed(2026101900+key+repeat*1000)
                            pattern=pattern_fn(len(images))
                            delta,powers=attack(link,pattern,state,budget)
                            torch.manual_seed(2026102000+key+repeat*1000)
                            random,_=attack(link,torch.randn(len(images),1408,64,device='cuda',dtype=torch.complex64),state,budget)
                            random=match_received_power(random,delta)
                            rp=10*torch.log10(random.abs().square().mean(-1)/state['power'][:,0])
                            torch.testing.assert_close(rp,powers['received_psr_db'],atol=1e-4,rtol=0)
                            clean_latent=link.decode(state['rx'],state)
                            attack_latent=link.decode(state['rx']+delta,state)
                            clean=score(images,p0.decode(clean_latent))
                            attacked=score(images,p0.decode(attack_latent))
                            random_score=score(images,p0.decode(link.decode(state['rx']+random,state)))
                            diagnostic={} if critic is None else {'critic_clean_correct':(critic(clean_latent)>0).float(),
                                'critic_attack_correct':(critic(attack_latent)<0).float()}
                            for i in range(len(images)):
                                rows.append({'key':key+i,'repeat':repeat,'snr_db':snr,'budget_db':budget,
                                    'clean':float(clean[i]),'attack':float(attacked[i]),'random':float(random_score[i]),
                                    'random_received_psr_db':float(rp[i]),
                                    **{k:float(v[i]) for k,v in powers.items()},**{k:float(v[i]) for k,v in diagnostic.items()}})
                            key+=len(images)
                            if args.smoke: break
                        close(data)
        return rows

    # CUDA equation sign and baseline regression before any training.
    sample=torch.tensor([[[3+4j,0]]],device='cuda')
    c=torch.zeros(1,device='cuda',requires_grad=True)
    a=torch.zeros(1,device='cuda',requires_grad=True)
    terms=generator_terms([torch.tensor(.5,device='cuda')],[c],[a],sample,torch.zeros_like(sample),'literal_norm')
    loss=-terms['rx']+terms['ds']-terms['dv']
    torch.testing.assert_close(loss,torch.tensor(-.5-2*__import__('math').log(2)-5,device='cuda'))
    if float(torch.autograd.grad(loss,a,retain_graph=True)[0])>=0: raise AssertionError('wrong G sign')
    if float(torch.autograd.grad(-terms['ds'],a)[0])<=0: raise AssertionError('wrong D sign')
    images=next(iter(DataLoader(valid,batch_size=2)))[0].to('cuda')
    with torch.no_grad():
        link=build_analog_link('cuda',10)
        state=link.prepare(p0.encode(images),bank=eval_bank)
        direct=score(images,p0.decode(state['direct_rx']))
        wrapped=score(images,p0.decode(link.decode(state['rx'],state)))
        torch.testing.assert_close(direct,wrapped,atol=2e-4,rtol=0)
    close(valid)
    save(args.output_dir/'equation_checks.json',{'minimax_signs':True,'complex_L1_norm':True,
        'paired_original_baseline':True,'max_paired_difference_db':float((direct-wrapped).abs().max())})
    results={}
    for profile in args.profiles:
        torch.manual_seed(args.seed)
        generator=build_full_generator(warm)
        critic=SignalDiscriminator().to('cuda')
        optimizer=torch.optim.Adam([{'params':generator.backbone.parameters(),'lr':2e-4},
            {'params':generator.coefficients.parameters(),'lr':2e-4},
            {'params':[generator.mean_iq],'lr':.02}, {'params':[generator.basis_iq],'lr':.005}],betas=(.5,.999))
        d_optimizer=torch.optim.Adam(critic.parameters(),lr=2e-4,betas=(.5,.999))
        with torch.no_grad():
            first=generator(torch.rand(2,100,device='cuda'))
            second=generator(torch.rand(2,100,device='cuda'))
            if first.shape!=(2,1408,64) or float((first-second).abs().mean())<=1e-8:
                raise AssertionError('wrong shape or z-independent generator')
            # Regression for the observed v1 failure: force a huge constant
            # coefficient head and require independent codes to remain distinct.
            original_weight=generator.coefficients.weight.detach().clone()
            original_bias=generator.coefficients.bias.detach().clone()
            generator.coefficients.weight.zero_()
            generator.coefficients.bias.fill_(1000.)
            stress_a=generator(torch.rand(2,100,device='cuda'))
            stress_b=generator(torch.rand(2,100,device='cuda'))
            relative_distance=float((stress_a-stress_b).abs().mean()/stress_a.abs().square().mean().sqrt())
            if relative_distance<.005: raise AssertionError('saturated head hid the random code')
            generator.coefficients.weight.copy_(original_weight)
            generator.coefficients.bias.copy_(original_bias)
            save(args.output_dir/(profile+'_latent_regression.json'),{'forced_head_bias':1000.,
                'relative_code_distance':relative_distance,'passed':True})
        loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        iterator=iter(loader)
        history=[]
        best=-float('inf')
        coefficients=(0.,0.)
        started=time.time()
        for step in range(1,args.steps+1):
            generator.train()
            try: images,_=next(iterator)
            except StopIteration:
                iterator=iter(loader)
                images,_=next(iterator)
            if args.smoke: images=images[:2]
            images=images.to('cuda')
            torch.manual_seed(args.seed+step*31)
            snr=[10,25,100][int(torch.randint(3,()).item())]
            budget=[-16,-12,-10][int(torch.randint(3,()).item())]
            probability=1. if profile=='literal' else permutation_probability(step,args.steps)
            pattern_a=generator(torch.rand(len(images),100,device='cuda'))
            pattern_b=generator(torch.rand(len(images),100,device='cuda'))
            link=build_analog_link('cuda',snr)
            inputs=[]
            rx_losses=[]
            for eot in range(args.eot_samples):
                for index,weight in enumerate(weights):
                    with torch.no_grad():
                        latent,shape=encode(index,images)
                        torch.manual_seed(args.seed+step*101+eot*10007)
                        state=link.prepare(latent,bank=train_bank)
                        use_permutation=torch.rand(len(images),device='cuda')<probability
                        state['permutation']=torch.where(use_permutation[:,None],state['permutation'],
                            torch.arange(64,device='cuda').expand(len(images),-1))
                        clean=link.decode(state['rx'],state)
                    attacks=[]
                    mses=[]
                    for pattern in [pattern_a,pattern_b]:
                        delta,_=attack(link,pattern,state,budget)
                        received=link.decode(state['rx']+delta,state)
                        attacks.append(received)
                        mses.append((decode(index,received,shape)-images).square().mean())
                    rx_losses.append(weight*torch.stack(mses).mean()/args.eot_samples)
                    inputs.append((weight/args.eot_samples,clean,attacks))
            # Alternating D update: attacked signals detached; no generator gradient.
            optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(True)
            dloss=sum(weight*(F.softplus(-critic(clean.detach())).mean()+
                torch.stack([F.softplus(critic(a.detach())).mean() for a in attacked]).mean())
                for weight,clean,attacked in inputs)
            d_optimizer.zero_grad(set_to_none=True)
            dloss.backward()
            if any(p.grad is not None for p in generator.parameters()): raise AssertionError('D update leaked into G')
            d_optimizer.step()
            d_optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            clean_logits=[]
            attack_logits=[]
            for weight,clean,attacked in inputs:
                for a in attacked:
                    # Each physical pair has two z samples; average their GAN term.
                    clean_logits.append((weight/2,critic(clean)))
                    attack_logits.append((weight/2,critic(a)))
            mode='literal_norm' if profile=='literal' else 'rms_normalized_mean'
            terms=generator_terms(rx_losses,[v for _,v in clean_logits],[v for _,v in attack_logits],pattern_a,pattern_b,mode)
            # Restore the same ensemble/EOT weights for Eq. 11 as for Eq. 9.
            terms['ds']=sum(weight*(F.logsigmoid(c).mean()+F.logsigmoid(-a).mean())
                            for (weight,c),(_,a) in zip(clean_logits,attack_logits))
            if step==1 or step%100==0 or step==args.steps:
                parameters=list(generator.parameters())
                norms={}
                for name,value in [('rx',-terms['rx']),('ds',terms['ds']),('dv',-terms['dv'])]:
                    grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                    norms[name]=float(sum(g.square().sum() for g in grads if g is not None).sqrt())
                proposed=balanced_coefficients(norms['rx'],norms['ds'],norms['dv'])
                coefficients=proposed if step==1 else tuple(.8*old+.2*new for old,new in zip(coefficients,proposed))
            ramp=min(1.,step/max(1,args.steps//4))
            beta_ds,beta_dv=(1.,1.) if profile=='literal' else tuple(ramp*c for c in coefficients)
            if profile=='regularizer_off': beta_ds,beta_dv=0.,0.
            gloss=-terms['rx']+beta_ds*terms['ds']-beta_dv*terms['dv']
            optimizer.zero_grad(set_to_none=True)
            gloss.backward()
            if not bool(torch.isfinite(gloss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in generator.parameters()):
                raise FloatingPointError('invalid G objective/gradient')
            if any(p.grad is not None for p in critic.parameters()): raise AssertionError('G update changed critic gradients')
            if any(p.grad is not None for model in codecs for p in model.parameters()): raise AssertionError('codec accumulated gradients')
            torch.nn.utils.clip_grad_norm_(generator.parameters(),10.)
            optimizer.step()
            record={'step':step,'snr_db':snr,'budget_db':budget,'permutation_probability':probability,
                'D_loss':float(dloss.detach()),'G_loss':float(gloss.detach()),
                **{key:float(value.detach()) for key,value in terms.items()},'beta_ds':beta_ds,'beta_dv':beta_dv}
            if step==1 or step%100==0 or step==args.steps:
                record['gradient_l2_unweighted']=norms
                record['gradient_l2_weighted']={'rx':norms['rx'],'ds':beta_ds*norms['ds'],'dv':beta_dv*norms['dv']}
                record['literal_raw_diversity']=float(diversity(pattern_a.detach(),pattern_b.detach()))
            history.append(record)
            if step%200==0 or step==args.steps:
                generator.eval()
                rows=evaluate(lambda b:generator(torch.rand(b,100,device='cuda')),valid,[10,100],[-16,-10],1)
                value=sum(r['clean']-r['attack'] for r in rows)/len(rows)
                if value>best:
                    best,best_step=value,step
                    best_g={k:v.detach().cpu().clone() for k,v in generator.state_dict().items()}
                    best_d={k:v.detach().cpu().clone() for k,v in critic.state_dict().items()}
                save(args.output_dir/(profile+'_history.json'),history)
                print(json.dumps({'stage':'training','profile':profile,'step':step,'P0_validation_drop_db':value,
                    'beta_ds':beta_ds,'beta_dv':beta_dv}),flush=True)
        generator.load_state_dict(best_g)
        generator.eval()
        critic.load_state_dict(best_d)
        critic.eval()
        path=args.output_dir/(profile+'_best.pt')
        torch.save({'generator':best_g,'critic':best_d,'best_step':best_step,'validation_drop_db':best,'profile':profile},path)
        with torch.no_grad():
            torch.manual_seed(2026102200)
            a=generator(torch.rand(8,100,device='cuda'))
            b=generator(torch.rand(8,100,device='cuda'))
            probe=generator_terms([torch.tensor(0.,device='cuda')],[torch.zeros(1,device='cuda')],
                [torch.zeros(1,device='cuda')],a,b,'rms_normalized_mean')
            diversity_probe={'normalized_mean_distance':float(probe['dv']),
                'raw_mean_distance':float((a-b).abs().mean()),'mean_pattern_rms':float(a.abs().square().mean().sqrt())}
            if diversity_probe['normalized_mean_distance']<.001:
                raise AssertionError('trained generator collapsed despite diversity mechanism')
        rows=evaluate(lambda b:generator(torch.rand(b,100,device='cuda')),test,
            [10] if args.smoke else [10,25,100],[-10] if args.smoke else [-20,-18,-16,-14,-12,-10],
            1 if args.smoke else args.evaluation_repeats,critic)
        results[profile]={'best_step':best_step,'P0_validation_drop_db':best,'history':history,
            'checkpoint_sha256':sha(path),'diversity_probe':diversity_probe,'rows':rows,'seconds':time.time()-started}
        save(args.output_dir/(profile+'_metrics.json'),results[profile])
        print(json.dumps({'stage':'profile_complete','profile':profile,'records':len(rows),'diversity':diversity_probe}),flush=True)
    results['warm_shared_reference']={'rows':evaluate(lambda b:torch.view_as_complex(warm).expand(b,-1,-1),test,
        [10] if args.smoke else [10,25,100],[-10] if args.smoke else [-20,-18,-16,-14,-12,-10],
        1 if args.smoke else args.evaluation_repeats),'checkpoint_sha256':sha(args.warm_checkpoint)}
    close(train)
    if fingerprints!=[state_fingerprint(model) for model in codecs]: raise AssertionError('codec changed')
    save(args.output_dir/'metrics.json',{'status':'completed','identity':identity,'profiles':results,
        'codecs_unchanged':True,'codec_parameter_gradients':False,
        'data_sha256':{'train':sha(train.path/'data.mdb'),'valid':sha(test.path/'data.mdb')}})
    print(json.dumps({'stage':'completed','records':sum(len(r['rows']) for r in results.values())}),flush=True)


if __name__=='__main__': main()
