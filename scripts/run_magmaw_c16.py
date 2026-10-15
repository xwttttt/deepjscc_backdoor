"""Server-only c16 P0 white-box universal OTA attack and explicit PGD reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--evaluation-seed',type=int,default=20261014)
    parser.add_argument('--profiles',nargs='+',default=['mse','psnr'])
    parser.add_argument('--evaluation-repeats',type=int,default=5)
    parser.add_argument('--pgd-reference',action='store_true')
    args=parser.parse_args()
    if any(x not in ['mse','psnr'] for x in args.profiles) or min(args.steps,args.batch_size,args.evaluation_repeats)<1:
        parser.error('valid profiles and positive budgets required')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import build_pgm,paper_attack
    from rf_backdoor.magmaw_effective import cap_received,match_received_power
    from rf_backdoor.magmaw_c16 import build_analog_link
    if not torch.cuda.is_available():
        raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    device='cuda:0'
    args.output_dir.mkdir(parents=True,exist_ok=False)
    checkpoint_hash=sha(args.checkpoint)
    if checkpoint_hash!='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542':
        raise ValueError('expected the original DIV2K c16 P0 checkpoint')
    adapter=DeepJSCCAdapter.from_local(c=16,snr_db=10)
    adapter.model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    adapter.model.to(device).eval().requires_grad_(False)
    before={name:parameter.detach().clone() for name,parameter in adapter.model.state_dict().items()}
    train,validation,test=load_div2k_splits(ROOT/'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',
                                          ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    max_symbols=1408  # Fixed public frame duration: ceil((ceil(65536/48)+1)/32)*32.
    train_bank=channel_bank(2000,args.seed+100,device)
    evaluation_bank=channel_bank(2000,args.evaluation_seed+200,device)
    identity={'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
              'checkpoint_sha256':checkpoint_hash,'c':16,'white_box':True,'new_codec_training':False,
              'legitimate_channel':'unit-gain AWGN, known channel, continuous analog OFDM; no QAM/no noisy CE',
              'attacker':'independent 8-tap synthetic Ha, phase/fractional delay/frequency permutation EOT',
              'power':'pre-Ha energy projection then offline received capture cap in both training and evaluation',
              'train_samples':len(train),'validation_keys':[0,50],'test_keys':[50,100],
              'fixed_attack_symbols':max_symbols,
              'code_sha256':{name:sha(ROOT/name) for name in ['scripts/run_magmaw_c16.py','src/rf_backdoor/magmaw_c16.py',
                  'src/rf_backdoor/magmaw_paper.py','src/rf_backdoor/magmaw_full.py','src/rf_backdoor/magmaw_effective.py',
                  'src/rf_backdoor/adapter.py','src/src/deepjscc/model.py','scripts/div2k_lmdb_dataset.py']}}
    save(args.output_dir/'identity.json',identity)
    def close(data):
        if data._env is not None:
            data._env.close()
            data._env=None
    def error(images,decoded):
        return (images-decoded).square().mean((1,2,3)).clamp_min(1e-12)
    def score(images,decoded):
        return -10*torch.log10(error(images,decoded))
    def status(**value):
        save(args.output_dir/'status.json',value)
        print(json.dumps(value),flush=True)
    def attack(link,pattern,state,budget):
        delta,powers=paper_attack(link,pattern,state,budget,max_symbols=max_symbols)
        powers['pre_cap_received_psr_db']=powers['received_psr_db']
        delta=cap_received(delta,state,budget)
        powers['received_psr_db']=10*torch.log10(delta.abs().square().mean(-1)/state['power'][:,0])
        if bool((powers['received_psr_db']>budget+1e-4).any()):
            raise AssertionError('received cap exceeded')
        return delta,powers
    def evaluate(generator,data,budgets,snrs,repeats):
        generator.eval()
        rows=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            for snr in snrs:
                link=build_analog_link(device,snr)
                for repeat in range(repeats):
                    for budget in budgets:
                        torch.manual_seed(args.evaluation_seed+1000+repeat)
                        key=data.start
                        for images,_ in DataLoader(data,batch_size=args.batch_size):
                            images=images.to(device)
                            latent=adapter.encode(images)
                            state=link.prepare(latent,bank=evaluation_bank)
                            pattern=generator(torch.rand(len(images),100,device=device))
                            delta,powers=attack(link,pattern,state,budget)
                            random_delta,_=attack(link,torch.randn_like(pattern),state,budget)
                            random_delta=match_received_power(random_delta,delta)
                            powers['random_received_psr_db']=10*torch.log10(random_delta.abs().square().mean(-1)/state['power'][:,0])
                            torch.testing.assert_close(powers['random_received_psr_db'],powers['received_psr_db'],atol=1e-4,rtol=0)
                            clean=score(images,adapter.decode(link.decode(state['rx'],state)))
                            attacked=score(images,adapter.decode(link.decode(state['rx']+delta,state)))
                            random_scores=score(images,adapter.decode(link.decode(state['rx']+random_delta,state)))
                            for i in range(len(images)):
                                rows.append({'key':key+i,'repeat':repeat,'snr_db':snr,'psr_budget_db':budget,
                                    'clean':float(clean[i]),'attack':float(attacked[i]),'random':float(random_scores[i]),
                                    'drop_db':float(clean[i]-attacked[i]),'random_drop_db':float(clean[i]-random_scores[i]),
                                    'advantage_db':float(random_scores[i]-attacked[i]),
                                    **{k:float(v[i]) for k,v in powers.items()}})
                            key+=len(images)
                        close(data)
        return rows
    # Paired mathematical baseline gate uses the SAME AWGN realization.
    gate={}
    images=next(iter(DataLoader(validation,batch_size=8)))[0].to(device)
    with torch.no_grad():
        latent=adapter.encode(images)
        assert latent.shape[1]==65536
        for snr in [10,25,100]:
            torch.manual_seed(args.evaluation_seed)
            link=build_analog_link(device,snr)
            state=link.prepare(latent,bank=evaluation_bank)
            restored=link.decode(state['clean'],state)
            torch.testing.assert_close(restored,latent,atol=2e-6,rtol=2e-6)
            received=link.decode(state['rx'],state)
            torch.testing.assert_close(received,state['direct_rx'],atol=2e-6,rtol=2e-6)
            direct=score(images,adapter.decode(state['direct_rx']))
            wrapped=score(images,adapter.decode(received))
            torch.testing.assert_close(direct,wrapped,atol=2e-4,rtol=0)
            gate[str(snr)]={'direct_psnr_db':float(direct.mean()),'ofdm_psnr_db':float(wrapped.mean()),
                            'max_paired_psnr_difference_db':float((direct-wrapped).abs().max())}
        if gate['10']['ofdm_psnr_db']<36 or gate['100']['ofdm_psnr_db']<39:
            raise RuntimeError('original high-quality P0 baseline was not preserved')
    close(validation)
    save(args.output_dir/'baseline_checks.json',{'passed':True,'first_validation_batch':gate,
         'mapping':'continuous OFDM identity; identical AWGN realization as direct P0',
         'tolerance_db':2e-4})
    results={}
    for profile in args.profiles:
        torch.manual_seed(args.seed)
        generator=build_pgm().to(device)
        optimizer=torch.optim.Adam(generator.parameters(),lr=2e-4,betas=(.5,.999))
        batches=DataLoader(train,batch_size=args.batch_size,shuffle=True)
        iterator=iter(batches)
        best=-float('inf')
        history=[]
        started=time.time()
        for step in range(1,args.steps+1):
            generator.train()
            try: images,_=next(iterator)
            except StopIteration:
                iterator=iter(batches)
                images,_=next(iterator)
            images=images.to(device)
            # EOT across noise levels: retain SNR10, also attack high-quality P0.
            snr=[10,25,100][int(torch.randint(3,()).item())]
            budget=[-16,-12,-10][int(torch.randint(3,()).item())]
            link=build_analog_link(device,snr)
            with torch.no_grad():
                latent=adapter.encode(images)
                state=link.prepare(latent,bank=train_bank)
            pattern=generator(torch.rand(len(images),100,device=device))
            delta,_=attack(link,pattern,state,budget)
            mse=error(images,adapter.decode(link.decode(state['rx']+delta,state)))
            loss=-mse.mean() if profile=='mse' else -(10*torch.log10(mse)).mean()
            optimizer.zero_grad()
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in generator.parameters()):
                raise FloatingPointError('nonfinite attack objective or gradient')
            optimizer.step()
            history.append({'step':step,'snr_db':snr,'budget':budget,'loss':float(loss.detach())})
            if step%200==0 or step==args.steps:
                valid=evaluate(generator,validation,[-16,-10],[10,100],1)
                drop=sum(row['drop_db'] for row in valid)/len(valid)
                if drop>best:
                    best=drop
                    torch.save({'generator':generator.state_dict(),'step':step,'validation_drop_db':drop},args.output_dir/(profile+'_best.pt'))
                status(stage='training',profile=profile,step=step,validation_drop_db=drop)
                save(args.output_dir/(profile+'_history.json'),history)
        selected=torch.load(args.output_dir/(profile+'_best.pt'),map_location=device,weights_only=True)
        generator.load_state_dict(selected['generator'])
        rows=evaluate(generator,test,[-20,-18,-16,-14,-12,-10],[10,25,100],args.evaluation_repeats)
        results[profile]={'best_step':selected['step'],'validation_drop_db':best,'history':history,'test':rows,
                          'seconds':time.time()-started,'checkpoint_sha256':sha(args.output_dir/(profile+'_best.pt'))}
        save(args.output_dir/(profile+'_metrics.json'),results[profile])
        status(stage='profile_completed',profile=profile,records=len(rows))
    pgd=[]
    if args.pgd_reference:
        # Explicit per-image, channel-aware upper-bound reference, NOT a UAP.
        with torch.random.fork_rng(devices=[0]):
            for budget in [-16,-10]:
                torch.manual_seed(args.evaluation_seed+1000)
                link=build_analog_link(device,10)
                key=test.start
                for images,_ in DataLoader(test,batch_size=args.batch_size):
                    images=images.to(device)
                    with torch.no_grad():
                        latent=adapter.encode(images)
                        state=link.prepare(latent,bank=evaluation_bank)
                        clean=score(images,adapter.decode(link.decode(state['rx'],state)))
                    iq=nn.Parameter(torch.randn(len(images),max_symbols,64,2,device=device)*.1)
                    optimizer=torch.optim.Adam([iq],lr=.05)
                    best_delta=None
                    best_error=torch.full((len(images),),-1.,device=device)
                    for iteration in range(80):
                        delta,powers=attack(link,torch.view_as_complex(iq),state,budget)
                        mse=error(images,adapter.decode(link.decode(state['rx']+delta,state)))
                        with torch.no_grad():
                            improved=mse>best_error
                            if best_delta is None:
                                best_delta=delta.detach().clone()
                            else:
                                best_delta[improved]=delta.detach()[improved]
                            best_error=torch.maximum(best_error,mse.detach())
                        optimizer.zero_grad()
                        (-mse.mean()).backward()
                        optimizer.step()
                    with torch.no_grad():
                        random_delta,_=attack(link,torch.randn(len(images),max_symbols,64,device=device,dtype=torch.complex64),state,budget)
                        random_delta=match_received_power(random_delta,best_delta)
                        attacked=score(images,adapter.decode(link.decode(state['rx']+best_delta,state)))
                        random_score=score(images,adapter.decode(link.decode(state['rx']+random_delta,state)))
                        power=10*torch.log10(best_delta.abs().square().mean(-1)/state['power'][:,0])
                        assert bool((power<=budget+1e-4).all())
                        for i in range(len(images)):
                            pgd.append({'key':key+i,'repeat':0,'snr_db':10,'psr_budget_db':budget,
                                'clean':float(clean[i]),'attack':float(attacked[i]),'random':float(random_score[i]),
                                'drop_db':float(clean[i]-attacked[i]),'received_psr_db':float(power[i]),
                                'iterations':80,'image_conditioned':True,'instantaneous_attacker_channel_known':True})
                    key+=len(images)
                close(test)
        save(args.output_dir/'pgd_reference.json',pgd)
    frozen=all(torch.equal(before[name],value) for name,value in adapter.model.state_dict().items())
    if not frozen:
        raise AssertionError('P0 weights changed')
    close(train)
    save(args.output_dir/'metrics.json',{'status':'completed','identity':identity,'profiles':results,
        'baseline_checks':gate,'p0_weights_unchanged':frozen,'pgd_reference':pgd,
        'data_sha256':{'train':sha(train.path/'data.mdb'),'valid':sha(test.path/'data.mdb')},
        'limitations':['white-box universal training on the frozen c16 P0; no black-box transfer claim',
          'known unit legitimate AWGN channel; no estimated CSI or real RF',
          'offline received-power cap uses simulated capture power',
          'PGD is per-image and instantaneous-channel-aware, explicitly an upper-bound reference'],
        'environment':{'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}})
    status(stage='completed')


if __name__=='__main__':
    main()
