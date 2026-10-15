"""Image-only Algorithm 1 reconstruction and controlled equation-level diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--profiles',nargs='+',default=['literal','mean_diversity','no_diversity','distortion_only','legacy_projection'])
    p.add_argument('--evaluation-repeats',type=int,default=2)
    args=p.parse_args()
    if min(args.steps,args.batch_size,args.evaluation_repeats)<1:
        p.error('steps, batch size and repeats must be positive')
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader,Subset
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.magmaw_full import ImageCodec,SignalDiscriminator,WaveformOFDM,channel_bank
    from rf_backdoor.magmaw_paper import build_pgm,build_link,game_losses,paper_attack,project_energy,diversity
    if not torch.cuda.is_available():
        raise RuntimeError('dedicated-server CUDA runtime required')
    device='cuda:0'
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    modes={'literal':(1.,1.,'norm','before_channel'),
           'mean_diversity':(1.,1.,'mean','before_channel'),
           'no_diversity':(1.,0.,'norm','before_channel'),
           'distortion_only':(0.,0.,'norm','before_channel'),
           'legacy_projection':(1.,1.,'norm','after_channel')}
    if any(name not in modes for name in args.profiles):
        p.error('unknown profile')
    identity={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in
                  [Path(__file__),ROOT/'src/rf_backdoor/magmaw_paper.py',ROOT/'src/rf_backdoor/magmaw_full.py']},
              'scope':'image branch of Algorithm 1; literal vs explicitly labeled ablations',
              'new_codec_training':False}
    save(args.output_dir/'identity.json',identity)
    train=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb')
    validation=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=0,stop=50)
    test=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    source=Subset(train,range(1600))
    def loader(data,shuffle=False):
        return DataLoader(data,batch_size=args.batch_size,shuffle=shuffle,num_workers=0)
    def close(data):
        if data._env is not None:
            data._env.close()
            data._env=None
    models={}
    codec_identity={}
    for c in [4,8]:
        for role in ['surrogate_small','surrogate_large','target']:
            name=f'{role}_c{c}'
            file=args.codec_dir/(name+'.pt')
            payload=torch.load(file,map_location='cpu',weights_only=True)
            model=ImageCodec(**payload['record']['spec']).to(device)
            model.load_state_dict(payload['model'],strict=True)
            models[name]=model.eval().requires_grad_(False)
            codec_identity[name]=sha(file)
    link=build_link(device,10)
    train_bank=channel_bank(2000,args.seed+100,device)
    evaluation_bank=channel_bank(2000,args.seed+200,device)
    def status(**value):
        save(args.output_dir/'status.json',value)
        print(json.dumps(value),flush=True)
    # CUDA analytical regression: hand-computed projection, minimax signs and norm.
    sample=torch.tensor([[3+4j,0]],device=device)
    torch.testing.assert_close(project_energy(sample,[1]),torch.tensor([[.6+.8j,0]],device=device))
    torch.testing.assert_close(diversity(sample,torch.zeros_like(sample)),torch.tensor(5.,device=device))
    clean_logits=torch.zeros(1,device=device,requires_grad=True)
    fake_logits=torch.zeros(1,device=device,requires_grad=True)
    losses=game_losses([torch.tensor(.5,device=device)],[clean_logits],[fake_logits],sample,torch.zeros_like(sample))
    torch.testing.assert_close(losses['generator'],torch.tensor(-.5-2*__import__('math').log(2)-5,device=device))
    assert float(torch.autograd.grad(losses['generator'],fake_logits,retain_graph=True)[0])<0
    assert float(torch.autograd.grad(losses['discriminator'],fake_logits)[0])>0
    save(args.output_dir/'equation_checks.json',{'projection':'passed','complex_l1':'passed','minimax_signs':'passed'})
    def scores(images,reconstruction):
        return -10*torch.log10((images-reconstruction).square().mean((1,2,3)).clamp_min(1e-12))
    def evaluate(generator,data,roles,projection,budgets,repeats):
        generator.eval()
        rows=[]
        with torch.random.fork_rng(devices=[0]):
            with torch.no_grad():
                for repeat in range(repeats):
                    for c in [4,8]:
                        for degree in [4,16,64]:
                            for budget in budgets:
                                torch.manual_seed(args.seed+1000+repeat)
                                key=data.start
                                for images,_ in loader(data):
                                    images=images.to(device)
                                    pgm=generator(torch.rand(len(images),100,device=device))
                                    for role in roles:
                                        model=models[f'{role}_c{c}']
                                        z,shape=model.encode(images)
                                        state=link.prepare(z,degree,evaluation_bank)
                                        delta,powers=paper_attack(link,pgm,state,budget,projection=projection)
                                        clean=scores(images,model.decode(link.decode(state['rx'],state),shape))
                                        attacked=scores(images,model.decode(link.decode(state['rx']+delta,state),shape))
                                        for i in range(len(images)):
                                            rows.append({'key':key+i,'repeat':repeat,'role':role,'c':c,'modulation':degree,
                                                'psr_budget_db':budget,'clean':float(clean[i]),'attack':float(attacked[i]),
                                                'drop_db':float(clean[i]-attacked[i]),
                                                **{k:float(v[i]) for k,v in powers.items()}})
                                    key+=len(images)
                                close(data)
        return rows
    results={}
    for name in args.profiles:
        beta_ds,beta_dv,reduction,projection=modes[name]
        torch.manual_seed(args.seed)
        generator=build_pgm().to(device)
        discriminator=SignalDiscriminator().to(device)
        optimizer=torch.optim.Adam(generator.parameters(),lr=2e-4,betas=(.5,.999))
        d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=2e-4,betas=(.5,.999))
        batches=loader(source,True)
        iterator=iter(batches)
        history=[]
        best=-float('inf')
        parameters=list(generator.parameters())
        start=time.time()
        for step in range(1,args.steps+1):
            generator.train()
            try: images,_=next(iterator)
            except StopIteration:
                iterator=iter(batches)
                images,_=next(iterator)
            images=images.to(device)
            c=[4,8][int(torch.randint(2,()).item())]
            degree=[4,16,64][int(torch.randint(3,()).item())]
            first=generator(torch.rand(len(images),100,device=device))
            second=generator(torch.rand(len(images),100,device=device))
            mse,clean_signals,attack_signals=[],[],[]
            for role in ['surrogate_small','surrogate_large']:
                model=models[f'{role}_c{c}']
                with torch.no_grad(): z,shape=model.encode(images)
                state=link.prepare(z,degree,train_bank)
                delta,_=paper_attack(link,first,state,-16,projection=projection)
                clean=link.decode(state['rx'],state)
                attacked=link.decode(state['rx']+delta,state)
                clean_signals.append(clean)
                attack_signals.append(attacked)
                mse.append(F.mse_loss(model.decode(attacked,shape),images))
            if beta_ds:
                discriminator.requires_grad_(True)
                dl=game_losses(mse,[discriminator(x.detach()) for x in clean_signals],
                               [discriminator(x.detach()) for x in attack_signals],first.detach(),second.detach())['discriminator']
                d_optimizer.zero_grad()
                dl.backward()
                d_optimizer.step()
            discriminator.requires_grad_(False)
            terms=game_losses(mse,[discriminator(x) for x in clean_signals],
                              [discriminator(x) for x in attack_signals],first,second,
                              beta_ds=beta_ds,beta_dv=beta_dv,diversity_reduction=reduction)
            record={'step':step,'c':c,'modulation':degree,
                    **{k:float(v.detach()) for k,v in terms.items()}}
            if step in [1,50,args.steps]:
                gradients={}
                weighted={'reconstruction':-terms['reconstruction'],
                          'discriminative':beta_ds*terms['discriminative'],
                          'diversity':-beta_dv*terms['diversity']}
                for term,value in weighted.items():
                    grad=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                    gradients[term]=float(sum(g.square().sum() for g in grad if g is not None).sqrt())
                record['gradient_l2']=gradients
            optimizer.zero_grad()
            terms['generator'].backward()
            if not bool(torch.isfinite(terms['generator'])):
                raise FloatingPointError('nonfinite objective')
            optimizer.step()
            history.append(record)
            if step%100==0 or step==args.steps:
                valid=evaluate(generator,validation,['surrogate_small','surrogate_large'],projection,[-16],1)
                drop=sum(r['drop_db'] for r in valid)/len(valid)
                if drop>best:
                    best=drop
                    torch.save({'generator':generator.state_dict(),'step':step,'validation_drop_db':drop},
                               args.output_dir/(name+'_best.pt'))
                save(args.output_dir/(name+'_history.json'),history)
                status(stage='training',profile=name,step=step,validation_drop_db=drop)
        selected=torch.load(args.output_dir/(name+'_best.pt'),map_location=device,weights_only=True)
        generator.load_state_dict(selected['generator'])
        rows=evaluate(generator,test,['target'],projection,[-20,-18,-16,-14,-12,-10],args.evaluation_repeats)
        results[name]={'beta_ds':beta_ds,'beta_dv':beta_dv,'diversity_reduction':reduction,'projection':projection,
                      'best_step':selected['step'],'validation_drop_db':best,'seconds':time.time()-start,
                      'history':history,'test':rows,'checkpoint_sha256':sha(args.output_dir/(name+'_best.pt'))}
        save(args.output_dir/(name+'_metrics.json'),results[name])
        status(stage='profile_completed',profile=name,records=len(rows))
    close(train)
    report={'status':'completed','identity':identity,'codec_sha256':codec_identity,'profiles':results,
            'limitations':['image-only','DIV2K','custom D','synthetic Ha','retained legacy QAM codecs, LS CE receiver mismatch',
                           'only 400 steps by default; diagnostic budget, not full convergence'],
            'environment':{'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()},
            'data_sha256':{'train':sha(train.path/'data.mdb'),'valid':sha(test.path/'data.mdb')}}
    save(args.output_dir/'metrics.json',report)
    status(stage='completed',profiles=len(results))


if __name__=='__main__':
    main()
