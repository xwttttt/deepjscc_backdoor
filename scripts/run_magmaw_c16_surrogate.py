"""Three-process independent-c16 fit -> proxy-only attacks -> held-out P0 transfer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def save(path,data):
    path=Path(path)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)


def close(data):
    if data._env is not None:
        data._env.close()
        data._env=None


def score(torch,images,reconstruction):
    return -10*torch.log10((images-reconstruction).square().mean((1,2,3)).clamp_min(1e-12))


def code_hashes():
    names=['scripts/run_magmaw_c16_surrogate.py','src/rf_backdoor/magmaw_c16_surrogate.py',
           'src/rf_backdoor/magmaw_c16.py','src/rf_backdoor/magmaw_full.py','src/rf_backdoor/magmaw_paper.py',
           'src/rf_backdoor/magmaw_effective.py','scripts/div2k_lmdb_dataset.py','src/src/deepjscc/model.py']
    return {name:sha(ROOT/name) for name in names}


def fit_proxies(args):
    import torch
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.magmaw_c16_surrogate import PROXY_SPECS,make_proxy,state_fingerprint
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir()
    save(output/'fit_identity.json',{'arguments':vars(args)|{'output_dir':str(output)},'code_sha256':code_hashes(),
         'target_accessed':False,'initialization':'random; no P0 checkpoint is supplied to this process'})
    train,valid,_=load_div2k_splits(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    records=[]
    for spec in PROXY_SPECS:
        model=make_proxy(spec,'cuda')
        initial=state_fingerprint(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        loader=DataLoader(train,batch_size=args.batch_size,shuffle=True)
        iterator=iter(loader)
        history=[]
        best=-float('inf')
        for step in range(1,args.proxy_steps+1):
            model.train()
            try: images,_=next(iterator)
            except StopIteration:
                iterator=iter(loader)
                images,_=next(iterator)
            images=images.to('cuda')
            latent,shape=model.encode(images)
            # The original P0's legitimate continuous AWGN distribution.
            received=latent+torch.randn_like(latent)*(latent.detach().abs().square().mean(-1,keepdim=True)*.1).sqrt()
            loss=(model.decode(received,shape)-images).square().mean()
            optimizer.zero_grad()
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite proxy loss')
            optimizer.step()
            if step%500==0 or step==args.proxy_steps:
                model.eval()
                with torch.random.fork_rng(devices=[0]),torch.no_grad():
                    torch.manual_seed(20261014)
                    scores=[]
                    for images,_ in DataLoader(valid,batch_size=args.batch_size):
                        images=images.to('cuda')
                        latent,shape=model.encode(images)
                        received=latent+torch.randn_like(latent)*(.1**.5)
                        scores.extend(score(torch,images,model.decode(received,shape)).tolist())
                    close(valid)
                value=sum(scores)/len(scores)
                history.append({'step':step,'validation_psnr_db':value})
                if value>best:
                    best=value
                    selected={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    best_step=step
                print(json.dumps({'stage':'fit','proxy':spec['name'],'step':step,'validation_psnr_db':value}),flush=True)
        model.load_state_dict(selected)
        record={'name':spec['name'],'spec':spec,'role':'surrogate','initialization':'random',
                'initial_state_sha256':initial,'final_state_sha256':state_fingerprint(model),
                'steps':args.proxy_steps,'best_step':best_step,'validation_psnr_db':best,'history':history}
        path=output/'models'/(spec['name']+'.pt')
        torch.save({'model':selected,'record':record},path)
        record['checkpoint_sha256']=sha(path)
        records.append(record)
    close(train)
    save(output/'proxy_metrics.json',{'status':'completed','proxies':records,'target_accessed':False,
        'data_sha256':{'train':sha(train.path/'data.mdb'),'valid':sha(valid.path/'data.mdb')}})


def proxy_states(torch,models,images,link,bank,state_seed):
    result=[]
    with torch.no_grad():
        for model in models:
            torch.manual_seed(state_seed)
            latent,shape=model.encode(images)
            result.append((link.prepare(latent,bank=bank),shape))
    return result


def proxy_rows(torch,models,data,generator,bank,budgets,snrs,batch_size):
    from torch.utils.data import DataLoader
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_c16_surrogate import received_attack
    rows=[]
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        generator.eval()
        for snr in snrs:
            link=build_analog_link('cuda',snr)
            for budget in budgets:
                key=data.start
                for images,_ in DataLoader(data,batch_size=batch_size):
                    images=images.to('cuda')
                    states=proxy_states(torch,models,images,link,bank,20262014+key)
                    torch.manual_seed(20271014+key)
                    pattern=generator(torch.rand(len(images),100,device='cuda'))
                    for index,(model,(state,shape)) in enumerate(zip(models,states)):
                        delta,_=received_attack(link,pattern,state,budget)
                        clean=score(torch,images,model.decode(link.decode(state['rx'],state),shape))
                        attacked=score(torch,images,model.decode(link.decode(state['rx']+delta,state),shape))
                        rows.extend({'key':key+i,'proxy':index,'snr_db':snr,'budget_db':budget,
                            'clean':float(clean[i]),'attack':float(attacked[i]),'drop_db':float(clean[i]-attacked[i])} for i in range(len(images)))
                    key+=len(images)
                close(data)
    return rows


def train_attacks(args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.magmaw_c16_surrogate import load_proxies,state_fingerprint,received_attack
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import build_pgm
    output=args.output_dir
    fit=json.loads((output/'fit_identity.json').read_text())
    if fit['code_sha256']!=code_hashes():
        raise ValueError('source changed after fitting')
    records=json.loads((output/'proxy_metrics.json').read_text())['proxies']
    if any(sha(output/'models'/(r['name']+'.pt'))!=r['checkpoint_sha256'] for r in records):
        raise ValueError('proxy checkpoint changed')
    models=load_proxies(output/'models',records,'cuda')
    before=[state_fingerprint(model) for model in models]
    train,valid,test=load_div2k_splits(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    training_bank=channel_bank(2000,args.seed+100,'cuda')
    evaluation_bank=channel_bank(2000,20261214,'cuda')
    torch.manual_seed(args.seed)
    generator=build_pgm().to('cuda')
    optimizer=torch.optim.Adam(generator.parameters(),lr=2e-4,betas=(.5,.999))
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True)
    iterator=iter(loader)
    history=[]
    best=-float('inf')
    for step in range(1,args.attack_steps+1):
        generator.train()
        try: images,_=next(iterator)
        except StopIteration:
            iterator=iter(loader)
            images,_=next(iterator)
        images=images.to('cuda')
        snr=[10,25,100][int(torch.randint(3,()).item())]
        budget=[-16,-12,-10][int(torch.randint(3,()).item())]
        state_seed=int(torch.randint(2**30,()).item())
        link=build_analog_link('cuda',snr)
        states=proxy_states(torch,models,images,link,training_bank,state_seed)
        pattern=generator(torch.rand(len(images),100,device='cuda'))
        terms=[]
        for model,(state,shape) in zip(models,states):
            delta,_=received_attack(link,pattern,state,budget)
            terms.append(score(torch,images,model.decode(link.decode(state['rx']+delta,state),shape)))
        # Minimize ensemble reconstruction PSNR; no target model is present.
        loss=torch.stack(terms).mean()
        optimizer.zero_grad()
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in generator.parameters()):
            raise FloatingPointError('nonfinite ensemble attack gradient')
        optimizer.step()
        if step%200==0 or step==args.attack_steps:
            rows=proxy_rows(torch,models,valid,generator,evaluation_bank,[-16,-10],[10,100],args.batch_size)
            value=sum(r['drop_db'] for r in rows)/len(rows)
            history.append({'step':step,'proxy_validation_drop_db':value})
            if value>best:
                best=value
                torch.save({'generator':generator.state_dict(),'step':step,'proxy_validation_drop_db':value},output/'universal_best.pt')
            save(output/'attack_history.json',history)
            print(json.dumps({'stage':'attack','step':step,'proxy_validation_drop_db':value}),flush=True)
    selected=torch.load(output/'universal_best.pt',map_location='cuda',weights_only=True)
    generator.load_state_dict(selected['generator'])
    test_rows=proxy_rows(torch,models,test,generator,evaluation_bank,[-16,-10],[10,100],args.batch_size)
    adaptive=[]
    adaptive_records=[]
    for budget in [-16,-10]:
        link=build_analog_link('cuda',10)
        key=50
        for images,_ in DataLoader(test,batch_size=args.batch_size):
            images=images.to('cuda')
            state_seed=20262014+key
            states=proxy_states(torch,models,images,link,evaluation_bank,state_seed)
            torch.manual_seed(20281014+key)
            iq=nn.Parameter(torch.randn(len(images),1408,64,2,device='cuda')*.1)
            optimizer=torch.optim.Adam([iq],lr=.05)
            best_objective=torch.full((len(images),),float('inf'),device='cuda')
            best_iq=iq.detach().clone()
            with torch.no_grad():
                clean=torch.stack([score(torch,images,m.decode(link.decode(s['rx'],s),shape)) for m,(s,shape) in zip(models,states)]).mean(0)
            for iteration in range(args.adaptive_steps):
                terms=[]
                for model,(state,shape) in zip(models,states):
                    delta,_=received_attack(link,torch.view_as_complex(iq),state,budget)
                    terms.append(score(torch,images,model.decode(link.decode(state['rx']+delta,state),shape)))
                objective=torch.stack(terms).mean(0)
                with torch.no_grad():
                    changed=objective<best_objective
                    best_iq[changed]=iq.detach()[changed]
                    best_objective=torch.minimum(best_objective,objective.detach())
                optimizer.zero_grad()
                objective.mean().backward()
                if not bool(torch.isfinite(objective).all()) or not bool(torch.isfinite(iq.grad).all()):
                    raise FloatingPointError('nonfinite adaptive proxy objective')
                optimizer.step()
            adaptive.append({'key':key,'budget_db':budget,'snr_db':10,'state_seed':state_seed,
                             'raw':torch.view_as_complex(best_iq).cpu(),'count':len(images)})
            for i in range(len(images)):
                adaptive_records.append({'key':key+i,'budget_db':budget,'snr_db':10,
                    'proxy_clean_db':float(clean[i]),'proxy_attack_db':float(best_objective[i]),
                    'proxy_drop_db':float(clean[i]-best_objective[i])})
            key+=len(images)
        close(test)
    torch.save(adaptive,output/'adaptive_raw.pt')
    if before!=[state_fingerprint(model) for model in models]:
        raise AssertionError('surrogate codec changed during attack')
    close(train)
    save(output/'attack_complete.json',{'status':'completed','completed_at_unix':time.time(),'target_accessed':False,
        'target_queries':0,'target_gradients':False,'proxy_weights_unchanged':True,'code_sha256':code_hashes(),
        'proxy_checkpoints':{r['name']:r['checkpoint_sha256'] for r in records},
        'universal_checkpoint_sha256':sha(output/'universal_best.pt'),'adaptive_raw_sha256':sha(output/'adaptive_raw.pt'),
        'best_step':selected['step'],'proxy_validation_drop_db':best,'proxy_universal_test':test_rows,
        'adaptive_proxy_test':adaptive_records,'adaptive_steps':args.adaptive_steps})


def final_evaluation(args):
    import torch
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_c16_surrogate import P0_SHA256,state_fingerprint,received_attack
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import build_pgm
    from rf_backdoor.magmaw_effective import match_received_power
    output=args.output_dir
    complete=json.loads((output/'attack_complete.json').read_text())
    if complete['status']!='completed' or complete['target_accessed'] or complete['target_queries']!=0 or complete['code_sha256']!=code_hashes():
        raise ValueError('requires completed proxy-only attack generation')
    if sha(output/'universal_best.pt')!=complete['universal_checkpoint_sha256'] or sha(output/'adaptive_raw.pt')!=complete['adaptive_raw_sha256']:
        raise ValueError('selected attacks changed before target evaluation')
    if sha(args.target_checkpoint)!=P0_SHA256:
        raise ValueError('expected the original c16 P0')
    target_loaded_at=time.time()
    if target_loaded_at<=complete['completed_at_unix']:
        raise AssertionError('target loaded before attack was finalized')
    target=DeepJSCCAdapter.from_local(c=16)
    target.model.load_state_dict(torch.load(args.target_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    target.model.to('cuda').eval().requires_grad_(False)
    before=state_fingerprint(target.model)
    generator=build_pgm().to('cuda').eval()
    generator.load_state_dict(torch.load(output/'universal_best.pt',map_location='cuda',weights_only=True)['generator'])
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    bank=channel_bank(2000,20261214,'cuda')
    rows=[]
    def row_batch(images,state,pattern,link,budget,method,key,repeat):
        delta,powers=received_attack(link,pattern,state,budget)
        random,_=received_attack(link,torch.randn_like(pattern),state,budget)
        random=match_received_power(random,delta)
        powers['random_received_psr_db']=10*torch.log10(random.abs().square().mean(-1)/state['power'][:,0])
        torch.testing.assert_close(powers['random_received_psr_db'],powers['received_psr_db'],atol=1e-4,rtol=0)
        clean=score(torch,images,target.decode(link.decode(state['rx'],state)))
        attacked=score(torch,images,target.decode(link.decode(state['rx']+delta,state)))
        random_score=score(torch,images,target.decode(link.decode(state['rx']+random,state)))
        for i in range(len(images)):
            rows.append({'method':method,'key':key+i,'repeat':repeat,'snr_db':link.snr_db,'budget_db':budget,
                'clean':float(clean[i]),'attack':float(attacked[i]),'random':float(random_score[i]),
                'drop_db':float(clean[i]-attacked[i]),'advantage_db':float(random_score[i]-attacked[i]),
                **{k:float(v[i]) for k,v in powers.items()}})
    with torch.no_grad():
        for snr in [10,25,100]:
            link=build_analog_link('cuda',snr)
            for budget in [-20,-18,-16,-14,-12,-10]:
                for repeat in range(args.evaluation_repeats):
                    key=50
                    for images,_ in DataLoader(dataset,batch_size=args.batch_size):
                        images=images.to('cuda')
                        torch.manual_seed(20262014+key+repeat*1000)
                        state=link.prepare(target.encode(images),bank=bank)
                        torch.manual_seed(20271014+key+repeat*1000)
                        pattern=generator(torch.rand(len(images),100,device='cuda'))
                        row_batch(images,state,pattern,link,budget,'surrogate_universal',key,repeat)
                        key+=len(images)
                    close(dataset)
        adaptive=torch.load(output/'adaptive_raw.pt',map_location='cpu',weights_only=True)
        for record in adaptive:
            key=record['key']
            images=torch.stack([dataset[key-50+i][0] for i in range(record['count'])]).to('cuda')
            link=build_analog_link('cuda',record['snr_db'])
            torch.manual_seed(record['state_seed'])
            state=link.prepare(target.encode(images),bank=bank)
            row_batch(images,state,record['raw'].to('cuda'),link,record['budget_db'],'surrogate_image_conditioned',key,0)
        close(dataset)
    frozen=state_fingerprint(target.model)==before and all(p.grad is None for p in target.model.parameters())
    if not frozen:
        raise AssertionError('target weight/gradient isolation failed')
    save(output/'metrics.json',{'status':'completed','rows':rows,'proxy_metrics':json.loads((output/'proxy_metrics.json').read_text()),
        'attack_complete':complete,'target_checkpoint_sha256':P0_SHA256,'target_weights_unchanged':True,'target_gradients':False,
        'target_loaded_at_unix':target_loaded_at,'arguments':vars(args)|{'output_dir':str(output),'target_checkpoint':str(args.target_checkpoint)},
        'code_sha256':code_hashes(),'limitations':['model-gradient transfer only; no target queries before final evaluation',
          'continuous OFDM with known unit legitimate AWGN; no real RF',
          'received-power cap/matching is an offline oracle',
          'adaptive transfer still uses the image, actual Ha and a fixed noise realization on surrogates']})
    print(json.dumps({'stage':'evaluate','records':len(rows),'target_weights_unchanged':True}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['fit','attack','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--target-checkpoint',type=Path)
    parser.add_argument('--proxy-steps',type=int,default=10000)
    parser.add_argument('--attack-steps',type=int,default=12000)
    parser.add_argument('--adaptive-steps',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--evaluation-repeats',type=int,default=5)
    parser.add_argument('--seed',type=int,default=1234)
    args=parser.parse_args()
    # Load the torch-free policy first so the phase guard is locally testable.
    spec=importlib.util.spec_from_file_location('surrogate_policy',ROOT/'src/rf_backdoor/magmaw_c16_surrogate.py')
    policy=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    try:
        policy.enforce_phase(args.stage,args.target_checkpoint)
    except ValueError as error:
        parser.error(str(error))
    if min(args.proxy_steps,args.attack_steps,args.adaptive_steps,args.batch_size,args.evaluation_repeats)<1:
        parser.error('positive budgets required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    {'fit':fit_proxies,'attack':train_attacks,'evaluate':final_evaluation}[args.stage](args)


if __name__=='__main__':
    main()
