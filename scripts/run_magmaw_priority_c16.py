"""Strength-first complete UAP: protected target PSNR, EOT and spatial coverage.

Pilot selection and both refinements finish BEFORE any test evaluation.
P0 is explicitly white-box; G's only inference input remains uniform random z.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


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
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pilot-steps',type=int,default=400)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eot-samples',type=int,default=4)
    p.add_argument('--evaluation-repeats',type=int,default=5)
    p.add_argument('--seeds',type=int,nargs='+',default=[1234,5678])
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if min(args.pilot_steps,args.steps,args.batch_size,args.eot_samples,args.evaluation_repeats)<1 or not args.seeds:
        p.error('positive sizes required')
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_complete import build_full_generator,generator_terms
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import SignalDiscriminator,channel_bank
    from rf_backdoor.magmaw_c16_surrogate import load_proxies,state_fingerprint
    from rf_backdoor.magmaw_paper import paper_attack
    from rf_backdoor.magmaw_effective import cap_received,match_received_power
    from rf_backdoor.magmaw_priority import protected_gradients,project_auxiliary_np
    if not torch.cuda.is_available(): raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.previous_root/'runs/full/metrics.json').read_text())
    old_identity=previous['identity']
    target_path=Path(old_identity['arguments']['checkpoint'])
    if sha(target_path)!=old_identity['P0_sha256']: raise ValueError('P0 identity mismatch')
    p0=DeepJSCCAdapter.from_local(c=16)
    p0.model.load_state_dict(torch.load(target_path,map_location='cpu',weights_only=True)['model'],strict=True)
    p0.model.to('cuda').eval().requires_grad_(False)
    proxy_dir=Path(old_identity['arguments']['proxy_dir'])
    records=json.loads((proxy_dir.parent/'proxy_metrics.json').read_text())['proxies']
    if {r['name']:sha(proxy_dir/(r['name']+'.pt')) for r in records}!=old_identity['proxy_sha256']:
        raise ValueError('proxy identity mismatch')
    proxies=load_proxies(proxy_dir,records,'cuda')
    models=[p0.model,*proxies]
    fingerprints=[state_fingerprint(m) for m in models]
    initial_path=args.previous_root/'runs/full/regularizer_off_best.pt'
    if sha(initial_path)!=previous['profiles']['regularizer_off']['checkpoint_sha256']:
        raise ValueError('warm generator identity mismatch')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    train,valid,test=load_div2k_splits(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    train_bank=channel_bank(2000,2026102401,'cuda')
    valid_bank=channel_bank(2000,2026102402,'cuda')
    test_bank=channel_bank(2000,2026102403,'cuda')
    sources=['scripts/run_magmaw_priority_c16.py','src/rf_backdoor/magmaw_priority.py',
        'src/rf_backdoor/magmaw_complete.py','src/rf_backdoor/magmaw_c16.py','src/rf_backdoor/magmaw_full.py',
        'src/rf_backdoor/magmaw_paper.py','src/rf_backdoor/magmaw_effective.py','src/rf_backdoor/magmaw_c16_surrogate.py',
        'src/rf_backdoor/adapter.py','src/src/deepjscc/model.py','scripts/div2k_lmdb_dataset.py']
    identity={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'P0_sha256':old_identity['P0_sha256'],'proxy_sha256':old_identity['proxy_sha256'],
        'initial_generator_sha256':sha(initial_path),'code_sha256':{name:sha(ROOT/name) for name in sources},
        'white_box':True,'generator_input':'Uniform z only; no image, latent or CSI input',
        'auxiliary_gradient_caps':{'proxy1':.05,'proxy2':.05,'ds':.01,'dv':.005,'coverage':.2},
        'primary':'mean per-image P0 PSNR, averaged across two z and physical EOT',
        'curriculum':'all carrier permutations/phase/delay/Ha remain full throughout',
        'selection_score':'.7*SNR10/PSR-10 validation drop + .3*four-condition mean validation drop',
        'data_keys':{'train':[0,3200],'validation':[0,50],'test':[50,100]},
        'limitations':['white-box c16 image-only; known unit legitimate AWGN; synthetic Ha; no RF',
            'receiver-power cap/matching remains an offline oracle',
            'projection protects raw-gradient alignment before Adam, not guaranteed nonlinear progress',
            'two refinement seeds start from the SAME validation-selected pilot, not independent initializations',
            'test benchmark keys have been used in prior experiments; no claim of a newly sealed dataset']}
    save(args.output_dir/'identity.json',identity)

    def encode(index,images):
        return (p0.encode(images),None) if index==0 else proxies[index-1].encode(images)

    def decode(index,received,shape):
        return p0.decode(received) if index==0 else proxies[index-1].decode(received,shape)

    def image_psnr(images,output):
        return -10*torch.log10((images-output).square().mean((1,2,3)).clamp_min(1e-12))

    def block_psnr(images,output):
        mse=F.avg_pool2d((images-output).square().mean(1,keepdim=True),32)
        return -10*torch.log10(mse.clamp_min(1e-12))

    def attack(link,pattern,state,budget):
        delta,powers=paper_attack(link,pattern,state,budget,max_symbols=1408)
        powers['pre_cap_received_psr_db']=powers['received_psr_db']
        delta=cap_received(delta,state,budget)
        powers['received_psr_db']=10*torch.log10(delta.abs().square().mean(-1)/state['power'][:,0])
        if not bool(torch.isfinite(delta).all()) or bool((powers['received_psr_db']>budget+1e-4).any()):
            raise AssertionError('nonfinite or over-budget perturbation')
        return delta,powers

    def evaluate(pattern_fn,data,bank,snrs,budgets,repeats):
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
                            torch.manual_seed(2026102500+key+repeat*1000)
                            state=link.prepare(p0.encode(images),bank=bank)
                            torch.manual_seed(2026102600+key+repeat*1000)
                            pattern=pattern_fn(len(images))
                            delta,powers=attack(link,pattern,state,budget)
                            torch.manual_seed(2026102700+key+repeat*1000)
                            random,_=attack(link,torch.randn(len(images),1408,64,device='cuda',dtype=torch.complex64),state,budget)
                            random=match_received_power(random,delta)
                            random_psr=10*torch.log10(random.abs().square().mean(-1)/state['power'][:,0])
                            torch.testing.assert_close(random_psr,powers['received_psr_db'],atol=1e-4,rtol=0)
                            clean_image=p0.decode(link.decode(state['rx'],state))
                            attack_image=p0.decode(link.decode(state['rx']+delta,state))
                            clean=image_psnr(images,clean_image)
                            attacked=image_psnr(images,attack_image)
                            random_score=image_psnr(images,p0.decode(link.decode(state['rx']+random,state)))
                            coverage=(block_psnr(images,clean_image)-block_psnr(images,attack_image)>=3).float().mean((1,2,3))
                            for i in range(len(images)):
                                rows.append({'key':key+i,'repeat':repeat,'snr_db':snr,'budget_db':budget,
                                    'clean':float(clean[i]),'attack':float(attacked[i]),'random':float(random_score[i]),
                                    'fraction_blocks_drop_ge_3db':float(coverage[i]),
                                    'random_received_psr_db':float(random_psr[i]),**{k:float(v[i]) for k,v in powers.items()}})
                            key+=len(images)
                            if args.smoke: break
                        close(data)
        return rows

    def validate(generator):
        generator.eval()
        rows=evaluate(lambda b:generator(torch.rand(b,100,device='cuda')),valid,valid_bank,[10,100],[-16,-10],1)
        focus=[r for r in rows if r['snr_db']==10 and r['budget_db']==-10]
        focused=sum(r['clean']-r['attack'] for r in focus)/len(focus)
        average=sum(r['clean']-r['attack'] for r in rows)/len(rows)
        return .7*focused+.3*average,focused,average

    # Hand-computed regression actually exercises the CUDA implementation.
    for primary,auxiliary in [([1.,0.],[-2.,3.]),([1.,0.],[3.,4.]),([0.,0.],[3.,4.])]:
        g,diag=protected_gradients([torch.tensor(primary,device='cuda')],[torch.tensor(auxiliary,device='cuda')],.2)
        expected=torch.tensor(project_auxiliary_np(primary,auxiliary,.2),device='cuda',dtype=torch.float32)
        torch.testing.assert_close(g[0],expected,atol=1e-6,rtol=0)
    images=next(iter(DataLoader(valid,batch_size=2)))[0].to('cuda')
    with torch.no_grad():
        link=build_analog_link('cuda',10)
        state=link.prepare(p0.encode(images),bank=valid_bank)
        torch.testing.assert_close(image_psnr(images,p0.decode(state['direct_rx'])),
            image_psnr(images,p0.decode(link.decode(state['rx'],state))),atol=2e-4,rtol=0)
    close(valid)
    save(args.output_dir/'regression_checks.json',{'CUDA_projection_vs_hand_oracle':True,'paired_original_baseline':True})

    def train_step(generator,critic,optimizer,d_optimizer,images,profile,step,seed,record_diagnostics):
        torch.manual_seed(seed+step*31)
        if float(torch.rand(()))<.7: snr,budget=10,-10
        else:
            snr=[10,25,100][int(torch.randint(3,()).item())]
            budget=[-16,-12,-10][int(torch.randint(3,()).item())]
        link=build_analog_link('cuda',snr)
        patterns=[generator(torch.rand(len(images),100,device='cuda')) for _ in range(2)]
        with torch.no_grad():
            encoded=[encode(index,images) for index in range(3)]
        model_losses=[[],[],[]]
        spatial=[]
        inputs=[]
        for eot in range(args.eot_samples):
            for index,(latent,shape) in enumerate(encoded):
                with torch.no_grad():
                    torch.manual_seed(seed+step*101+eot*10007)
                    state=link.prepare(latent,bank=train_bank)
                    clean=link.decode(state['rx'],state)
                received=[]
                for pattern in patterns:
                    delta,_=attack(link,pattern,state,budget)
                    attacked=link.decode(state['rx']+delta,state)
                    output=decode(index,attacked,shape)
                    received.append(attacked)
                    model_losses[index].append(image_psnr(images,output).mean())
                    if index==0: spatial.append(block_psnr(images,output).mean())
                inputs.append((index,clean,received))
        optimizer.zero_grad(set_to_none=True)
        critic.requires_grad_(True)
        dloss=sum((1. if index==0 else .25)*(F.softplus(-critic(clean.detach())).mean()+
            torch.stack([F.softplus(critic(a.detach())).mean() for a in attacked]).mean())/args.eot_samples
            for index,clean,attacked in inputs)
        d_optimizer.zero_grad(set_to_none=True)
        dloss.backward()
        if any(p.grad is not None for p in generator.parameters()): raise AssertionError('D update leaked into G')
        d_optimizer.step()
        d_optimizer.zero_grad(set_to_none=True)
        critic.requires_grad_(False)
        ds=sum((1. if index==0 else .25)*(F.logsigmoid(critic(clean)).mean()+
            torch.stack([F.logsigmoid(-critic(a)).mean() for a in attacked]).mean())/args.eot_samples
            for index,clean,attacked in inputs)
        dummy=torch.zeros(1,device='cuda')
        diversity=generator_terms([dummy],[dummy],[dummy],patterns[0],patterns[1],'rms_normalized_mean')['dv']
        primary_loss=torch.stack(model_losses[0]).mean()
        terms={'proxy1':torch.stack(model_losses[1]).mean(),'proxy2':torch.stack(model_losses[2]).mean(),
               'ds':ds,'dv':-diversity,'coverage':torch.stack(spatial).mean()}
        parameters=list(generator.parameters())
        def gradient(value):
            raw=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
            return [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,raw)]
        primary=gradient(primary_loss)
        combined=[g.clone() for g in primary]
        diagnostics={}
        names=[] if profile=='target_only' else ['proxy1','proxy2','ds','dv']
        if profile=='coverage': names.append('coverage')
        for name in names:
            auxiliary=gradient(terms[name])
            protected,diag=protected_gradients(primary,auxiliary,identity['auxiliary_gradient_caps'][name])
            combined=[g+a for g,a in zip(combined,protected)]
            if record_diagnostics:
                diagnostics[name]={key:float(value) for key,value in diag.items()}
                if diagnostics[name]['projected_dot'] < -1e-5*max(1e-12,diagnostics[name]['primary_l2']**2):
                    raise AssertionError('opposing auxiliary survived projection')
                if diagnostics[name]['applied_aux_l2']>identity['auxiliary_gradient_caps'][name]*diagnostics[name]['primary_l2']+1e-5:
                    raise AssertionError('auxiliary gradient exceeded its cap')
        if not bool(torch.isfinite(primary_loss)) or any(not bool(torch.isfinite(g).all()) for g in combined):
            raise FloatingPointError('invalid strength gradient')
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(parameters,combined): p.grad=g
        torch.nn.utils.clip_grad_norm_(parameters,10.)
        optimizer.step()
        if any(p.grad is not None for m in models for p in m.parameters()): raise AssertionError('codec accumulated gradients')
        if any(p.grad is not None for p in critic.parameters()): raise AssertionError('G update leaked into D')
        return {'step':step,'snr_db':snr,'budget_db':budget,'primary_psnr':float(primary_loss.detach()),
            'ds':float(ds.detach()),'diversity':float(diversity.detach()),
            'spatial_psnr':float(terms['coverage'].detach()),'gradient_diagnostics':diagnostics}

    def train_run(name,profile,steps,seed,payload):
        directory=args.output_dir/name
        directory.mkdir()
        torch.manual_seed(seed)
        generator=build_full_generator(payload['generator']['mean_iq'].to('cuda'))
        generator.load_state_dict(payload['generator'],strict=True)
        critic=SignalDiscriminator().to('cuda')
        critic.load_state_dict(payload['critic'],strict=True)
        optimizer=torch.optim.Adam([{'params':generator.backbone.parameters(),'lr':2e-4},
            {'params':generator.coefficients.parameters(),'lr':2e-4},{'params':[generator.mean_iq],'lr':.02},
            {'params':[generator.basis_iq],'lr':.005}],betas=(.5,.999))
        initial_lrs=[group['lr'] for group in optimizer.param_groups]
        d_optimizer=torch.optim.Adam(critic.parameters(),lr=2e-4,betas=(.5,.999))
        loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
        iterator=iter(loader)
        best,focus,average=validate(generator)
        selected={key:value.detach().cpu().clone() for key,value in generator.state_dict().items()}
        selected_d={key:value.detach().cpu().clone() for key,value in critic.state_dict().items()}
        best_step=0
        history=[]
        validations=[{'step':0,'score':best,'focused_drop_db':focus,'mean_drop_db':average}]
        stale=0
        started=time.time()
        for step in range(1,steps+1):
            generator.train()
            try: images,_=next(iterator)
            except StopIteration:
                iterator=iter(loader)
                images,_=next(iterator)
            if args.smoke: images=images[:2]
            decay=.2+.8*(1+math.cos(math.pi*(step-1)/max(steps,1)))/2
            for group,lr in zip(optimizer.param_groups,initial_lrs): group['lr']=lr*decay
            record=train_step(generator,critic,optimizer,d_optimizer,images.to('cuda'),profile,step,seed,
                              step==1 or step%100==0 or step==steps)
            history.append(record)
            if step%200==0 or step==steps:
                value,focus,average=validate(generator)
                validations.append({'step':step,'score':value,'focused_drop_db':focus,'mean_drop_db':average})
                if value>best:
                    meaningful=value>best+.01
                    best,best_step=value,step
                    selected={key:value.detach().cpu().clone() for key,value in generator.state_dict().items()}
                    selected_d={key:value.detach().cpu().clone() for key,value in critic.state_dict().items()}
                    stale=0 if meaningful else stale+1
                else: stale+=1
                save(directory/'history.json',history)
                save(directory/'validation.json',validations)
                print(json.dumps({'stage':name,'profile':profile,'step':step,'validation_score':value,
                    'focused_drop_db':focus,'mean_drop_db':average}),flush=True)
                if step>=1600 and stale>=4: break
        path=directory/'best.pt'
        torch.save({'generator':selected,'critic':selected_d,'profile':profile,'seed':seed,'best_step':best_step},path)
        result={'name':name,'profile':profile,'seed':seed,'steps_executed':len(history),'best_step':best_step,
                'validation_score':best,'checkpoint_sha256':sha(path),'validations':validations,
                'seconds':time.time()-started}
        save(directory/'training.json',result)
        return result

    pilot={}
    candidates=['coverage'] if args.smoke else ['target_only','protected','coverage']
    for profile in candidates:
        pilot[profile]=train_run('pilot_'+profile,profile,args.pilot_steps,args.seeds[0],initial)
    selected_profile=max(pilot,key=lambda name:pilot[name]['validation_score'])
    selected_path=args.output_dir/('pilot_'+selected_profile)/'best.pt'
    save(args.output_dir/'pilot_selection.json',{'selected_profile':selected_profile,'candidates':pilot,
        'selection':'validation only; test not evaluated','selected_sha256':sha(selected_path)})
    payload=torch.load(selected_path,map_location='cpu',weights_only=True)
    refinements={}
    for seed in args.seeds:
        name='refine_'+str(seed)
        refinements[name]=train_run(name,selected_profile,args.steps,seed,payload)
    selected_name=max(refinements,key=lambda name:refinements[name]['validation_score'])
    locked={'status':'completed','completed_at_unix':time.time(),'selected_profile':selected_profile,
        'selected_refinement':selected_name,'refinements':refinements,'source_sha256':identity['code_sha256'],
        'selection':'validation only; all refinement files finalized before any test scoring'}
    save(args.output_dir/'training_complete.json',locked)
    results={}
    evaluated_at=time.time()
    for name,record in refinements.items():
        path=args.output_dir/name/'best.pt'
        if sha(path)!=record['checkpoint_sha256']: raise ValueError('locked candidate changed')
        payload=torch.load(path,map_location='cuda',weights_only=True)
        generator=build_full_generator(payload['generator']['mean_iq']).eval().requires_grad_(False)
        generator.load_state_dict(payload['generator'],strict=True)
        with torch.no_grad():
            torch.manual_seed(2026102800)
            a=generator(torch.rand(8,100,device='cuda'))
            b=generator(torch.rand(8,100,device='cuda'))
            distance=float((a/a.abs().square().mean((1,2),keepdim=True).sqrt()-
                            b/b.abs().square().mean((1,2),keepdim=True).sqrt()).abs().mean())
            if distance<.001: raise AssertionError('random code collapsed')
        results[name]={'training':record,'diversity_distance':distance,'rows':evaluate(
            lambda batch:generator(torch.rand(batch,100,device='cuda')),test,test_bank,
            [10] if args.smoke else [10,25,100],[-10] if args.smoke else [-20,-18,-16,-14,-12,-10],
            1 if args.smoke else args.evaluation_repeats)}
    for profile in ['regularizer_off','balanced']:
        path=args.previous_root/'runs/full'/(profile+'_best.pt')
        if sha(path)!=previous['profiles'][profile]['checkpoint_sha256']: raise ValueError('reference changed')
        payload=torch.load(path,map_location='cuda',weights_only=True)
        generator=build_full_generator(payload['generator']['mean_iq']).eval().requires_grad_(False)
        generator.load_state_dict(payload['generator'],strict=True)
        results['previous_'+profile]={'checkpoint_sha256':sha(path),'rows':evaluate(
            lambda batch:generator(torch.rand(batch,100,device='cuda')),test,test_bank,
            [10] if args.smoke else [10,25,100],[-10] if args.smoke else [-20,-18,-16,-14,-12,-10],
            1 if args.smoke else args.evaluation_repeats)}
    warm_path=Path(old_identity['arguments']['warm_checkpoint'])
    if sha(warm_path)!=old_identity['warm_checkpoint_sha256']: raise ValueError('shared reference changed')
    warm=torch.load(warm_path,map_location='cuda',weights_only=True)['raw_iq']
    results['shared_reference']={'checkpoint_sha256':sha(warm_path),'rows':evaluate(
        lambda batch:torch.view_as_complex(warm).expand(batch,-1,-1),test,test_bank,
        [10] if args.smoke else [10,25,100],[-10] if args.smoke else [-20,-18,-16,-14,-12,-10],
        1 if args.smoke else args.evaluation_repeats)}
    close(train)
    if fingerprints!=[state_fingerprint(m) for m in models]: raise AssertionError('codec weights changed')
    save(args.output_dir/'metrics.json',{'status':'completed','identity':identity,'training_complete':locked,
        'test_evaluation_started_at_unix':evaluated_at,'profiles':results,'codecs_unchanged':True,
        'codec_gradients':False,'data_sha256':{'train':sha(train.path/'data.mdb'),'valid':sha(test.path/'data.mdb')}})
    print(json.dumps({'stage':'completed','selected_profile':selected_profile,'selected_refinement':selected_name,
        'records':sum(len(v['rows']) for v in results.values())}),flush=True)


if __name__=='__main__': main()
