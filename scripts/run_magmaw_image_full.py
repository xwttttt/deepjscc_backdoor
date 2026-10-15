"""Full DIV2K image adaptation of Magmaw: codecs -> UAPs -> held-out transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/magmaw_image_full.toml')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='server-only bounded end-to-end contract check')
    args = p.parse_args()
    cfg = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        cfg.update(codec_steps=2, attack_epochs=1, evaluation_repeats=1, batch_size=2,
                   modulations=[16], rates_c=[4], psr_db=[-16], train_channel_count=16, test_channel_count=16)
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.magmaw_full import (ImageCodec, ResidualPGM, SignalDiscriminator,
                                       WaveformOFDM, channel_bank)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('dedicated-server CUDA environment required')
    device = cfg['device']
    output = args.output_dir.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output/'identity.json').exists():
        raise ValueError('resume requires an existing experiment identity')
    code_paths = [Path(__file__), ROOT/'src/rf_backdoor/magmaw_full.py',
                  ROOT/'src/src/deepjscc/model.py', ROOT/'scripts/div2k_lmdb_dataset.py']
    identity = {'config': cfg, 'code_sha256': {str(x.relative_to(ROOT)): sha(x) for x in code_paths},
                'config_sha256': sha(args.config)}
    if args.resume:
        if json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('refusing resume after config/source identity changed')
    else:
        write(output/'identity.json', identity)
    train = PreparedDIV2KLMDB(ROOT/cfg['train_lmdb'])
    valid = PreparedDIV2KLMDB(ROOT/cfg['valid_lmdb'], start=0, stop=50)
    test = PreparedDIV2KLMDB(ROOT/cfg['valid_lmdb'], start=50, stop=100)
    # Strict disjoint source keys for surrogate/attacker vs victim fitting.
    source = Subset(train, range(0, 1600))
    victim_source = Subset(train, range(1600, 3200))
    if args.smoke:
        source, victim_source = Subset(train, range(2)), Subset(train, range(1600,1602))
        valid = Subset(valid, range(2))
        test = Subset(test, range(2))
    def loader(data, shuffle=False):
        return DataLoader(data, batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0)
    def close(data):
        while isinstance(data, Subset):
            data = data.dataset
        if data._env is not None:
            data._env.close()
            data._env = None
    link = WaveformOFDM(device, cfg['snr_db'])
    train_bank = channel_bank(cfg['train_channel_count'], cfg['seed']+100, device)
    test_bank = channel_bank(cfg['test_channel_count'], cfg['seed']+200, device)
    torch.save({'training': train_bank.cpu(), 'evaluation': test_bank.cpu()}, output/'channel_banks.pt')
    def seed(value):
        torch.manual_seed(value)
    def psnr(image, reconstruction):
        return -10*torch.log10((image-reconstruction).square().mean((1,2,3)).clamp_min(1e-12))
    def status(stage, **details):
        value = {'stage': stage, 'updated_unix': time.time(), **details}
        write(output/'status.json', value)
        print(json.dumps(value), flush=True)
    def validate_codec(model):
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed(cfg['seed']+10)
            values = []
            with torch.no_grad():
                for images,_ in loader(valid):
                    images = images.to(device)
                    z, shape = model.encode(images)
                    state = link.prepare(z, 16, test_bank)
                    values.extend(psnr(images,model.decode(link.decode(state['rx'],state),shape)).tolist())
            close(valid)
        return sum(values)/len(values)
    models, codec_records = {}, []
    for c in cfg['rates_c']:
        for role, depth, width in [('surrogate_small',6,28),('surrogate_large',10,36),('target',5,32)]:
            name = f'{role}_c{c}'
            seed(cfg['seed']+c*10+depth)
            model = ImageCodec(c,depth,width).to(device)
            checkpoint = output/(name+'.pt')
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(payload['model'])
                record = payload['record']
                if not record.get('complete'):
                    raise RuntimeError('partial codec checkpoint is not a complete stage')
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate_codec'])
                batches = loader(victim_source if role=='target' else source, True)
                iterator = iter(batches)
                best = -float('inf')
                history = []
                for step in range(1,cfg['codec_steps']+1):
                    model.train()
                    try:
                        images,_ = next(iterator)
                    except StopIteration:
                        iterator = iter(batches)
                        images,_ = next(iterator)
                    images = images.to(device)
                    degree = cfg['modulations'][int(torch.randint(len(cfg['modulations']),()).item())]
                    z, shape = model.encode(images)
                    state = link.prepare(z,degree,train_bank)
                    reconstruction = model.decode(link.decode(state['rx'],state),shape)
                    loss = F.mse_loss(reconstruction,images)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if step % 250 == 0 or step==cfg['codec_steps']:
                        value = validate_codec(model)
                        history.append({'step': step, 'validation_psnr_db': value})
                        if value > best:
                            best = value
                            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        status('fit_codec', codec=name, step=step, validation_psnr_db=value)
                record = {'name':name,'spec':model.spec,'best_validation_psnr_db':best,
                          'steps':cfg['codec_steps'],'history':history,'complete':True}
                model.load_state_dict(best_state)
                torch.save({'model':best_state,'record':record},checkpoint)
            model.eval().requires_grad_(False)
            models[name] = model
            codec_records.append(record | {'checkpoint_sha256':sha(checkpoint)})
    close(train)
    write(output/'codec_metrics.json',codec_records)
    generators = {}
    def pattern(method, generator, b):
        if method=='magmaw':
            return generator(torch.rand(b,100,device=device))
        return generator.expand(b,-1,-1)
    def validation_attack(method, generator):
        if method=='magmaw':
            generator.eval()
        scores = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed(cfg['seed']+20)
            with torch.no_grad():
                for images,_ in loader(valid):
                    images=images.to(device)
                    for c in cfg['rates_c']:
                        # Surrogate validation only: target never used to select attack.
                        model=models[f'surrogate_small_c{c}']
                        z,shape=model.encode(images)
                        state=link.prepare(z,16,test_bank)
                        delta,_=link.attack(pattern(method,generator,len(images)),state,cfg['training_psr_db'])
                        clean=model.decode(link.decode(state['rx'],state),shape)
                        attacked=model.decode(link.decode(state['rx']+delta,state),shape)
                        scores.extend((psnr(images,clean)-psnr(images,attacked)).tolist())
            close(valid)
        return sum(scores)/len(scores)
    for method in cfg['attack_methods']:
        seed(cfg['seed']+30)
        generator = ResidualPGM().to(device) if method=='magmaw' else torch.nn.Parameter(
            torch.randn(1,32,64,dtype=torch.complex64,device=device)*.1)
        checkpoint = output/(method+'_best.pt')
        complete = output/(method+'_training.json')
        if complete.exists():
            payload=torch.load(checkpoint,map_location=device,weights_only=True)
            if method=='magmaw': generator.load_state_dict(payload['generator'])
            else: generator.data.copy_(payload['generator'])
            generators[method]=generator
            continue
        discriminator=SignalDiscriminator().to(device)
        parameters=generator.parameters() if method=='magmaw' else [generator]
        optimizer=torch.optim.Adam(parameters,lr=cfg['learning_rate_attack'],betas=(.5,.999))
        d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=cfg['learning_rate_attack'],betas=(.5,.999))
        best=-float('inf')
        history=[]
        started=time.time()
        for epoch in range(1,cfg['attack_epochs']+1):
            totals={'mse':0.,'d_loss':0.,'g_discriminator':0.,'diversity':0.,'batches':0}
            if method=='magmaw': generator.train()
            for images,_ in loader(source,True):
                images=images.to(device)
                c=cfg['rates_c'][int(torch.randint(len(cfg['rates_c']),()).item())]
                degree=cfg['modulations'][int(torch.randint(len(cfg['modulations']),()).item())]
                perturbation=pattern(method,generator,len(images))
                clean_list, attacked_list, mse_list=[],[],[]
                transform={'magmaw':'full','vanilla_uap':'none','sync_free_uap':'sync_free'}[method]
                for role in ['surrogate_small','surrogate_large']:
                    model=models[f'{role}_c{c}']
                    with torch.no_grad(): z,shape=model.encode(images)
                    state=link.prepare(z,degree,train_bank)
                    delta,_=link.attack(perturbation,state,cfg['training_psr_db'],transform=transform)
                    clean=link.decode(state['rx'],state)
                    attacked=link.decode(state['rx']+delta,state)
                    clean_list.append(clean)
                    attacked_list.append(attacked)
                    mse_list.append(F.mse_loss(model.decode(attacked,shape),images))
                distortion=torch.stack(mse_list).sum()
                d_loss=distortion.new_zeros(())
                g_stealth=distortion.new_zeros(())
                diversity=distortion.new_zeros(())
                if method=='magmaw':
                    discriminator.requires_grad_(True)
                    d_loss=sum(F.binary_cross_entropy_with_logits(discriminator(clean.detach()),torch.ones(len(images),device=device))+
                        F.binary_cross_entropy_with_logits(discriminator(attacked.detach()),torch.zeros(len(images),device=device))
                        for clean,attacked in zip(clean_list,attacked_list))/2
                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()
                    discriminator.requires_grad_(False)
                    # Exact saturating minimax generator term log(1-D(fake));
                    # D maximizes log D(clean)+log(1-D(fake)).
                    g_stealth=sum(F.logsigmoid(-discriminator(a)).mean() for a in attacked_list)/2
                    second=generator(torch.rand(len(images),100,device=device))
                    diversity=torch.view_as_real(perturbation-second).abs().mean()
                objective=-distortion+cfg['beta_discriminator']*g_stealth-cfg['beta_diversity']*diversity
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters() if method=='magmaw' else [generator],10)
                optimizer.step()
                for key,value in [('mse',distortion),('d_loss',d_loss),('g_discriminator',g_stealth),('diversity',diversity)]:
                    totals[key]+=float(value.detach())
                totals['batches']+=1
            validation=validation_attack(method,generator)
            totals={k:v/totals['batches'] if k!='batches' else v for k,v in totals.items()}
            history.append({'epoch':epoch,'validation_drop_db':validation,**totals})
            if validation>best:
                best=validation
                value=generator.state_dict() if method=='magmaw' else generator.detach()
                torch.save({'generator':value,'epoch':epoch,'validation_drop_db':validation},checkpoint)
            torch.save({'generator':generator.state_dict() if method=='magmaw' else generator.detach(),
                'optimizer':optimizer.state_dict(),'discriminator':discriminator.state_dict(),
                'd_optimizer':d_optimizer.state_dict(),'epoch':epoch},output/(method+'_last.pt'))
            write(output/(method+'_history.json'),history)
            status('fit_attack',method=method,epoch=epoch,epochs=cfg['attack_epochs'],validation_drop_db=validation)
        payload=torch.load(checkpoint,map_location=device,weights_only=True)
        if method=='magmaw': generator.load_state_dict(payload['generator'])
        else: generator.data.copy_(payload['generator'])
        write(complete,{'complete':True,'history':history,'seconds':time.time()-started,
                       'best_epoch':payload['epoch'],'best_sha256':sha(checkpoint)})
        generators[method]=generator
    close(train)
    # Same source frames, victim channel, AWGN and attacker transforms for each method.
    rows=[]
    for generator in generators.values():
        if isinstance(generator,torch.nn.Module): generator.eval()
    for c in cfg['rates_c']:
        model=models[f'target_c{c}']
        for degree in cfg['modulations']:
            for budget in cfg['psr_db']:
                condition=[]
                with torch.no_grad():
                    for repeat in range(cfg['evaluation_repeats']):
                        seed(cfg['seed']+300+repeat)
                        index=50
                        for images,_ in loader(test):
                            images=images.to(device)
                            z,shape=model.encode(images)
                            state=link.prepare(z,degree,test_bank)
                            clean=psnr(images,model.decode(link.decode(state['rx'],state),shape))
                            outputs={'clean':clean}
                            budgets={}
                            random,actual=link.random_attack(state,budget)
                            outputs['random']=psnr(images,model.decode(link.decode(state['rx']+random,state),shape))
                            budgets['random']=actual
                            for method,generator in generators.items():
                                delta,actual=link.attack(pattern(method,generator,len(images)),state,budget)
                                outputs[method]=psnr(images,model.decode(link.decode(state['rx']+delta,state),shape))
                                budgets[method]=actual
                            for i in range(len(images)):
                                record={'key':index+i,'repeat':repeat,'c':c,'coding_rate':c/48,
                                        'modulation':degree,'psr_db':budget,
                                        **{k:float(v[i]) for k,v in outputs.items()},
                                        'actual_psr_db':{k:float(v[i]) for k,v in budgets.items()}}
                                condition.append(record)
                            index+=len(images)
                        close(test)
                rows.extend(condition)
                write(output/'test_samples.json',rows)
                means={k:sum(r[k] for r in condition)/len(condition) for k in outputs}
                status('evaluate',c=c,modulation=degree,psr_db=budget,means=means)
    report={'status':'completed','identity':identity,'codec_records':codec_records,
            'channel_banks_sha256':sha(output/'channel_banks.pt'),'test':rows,
            'data_sha256':{'train':sha(train.path/'data.mdb'),
                           'valid':sha(ROOT/cfg['valid_lmdb']/'data.mdb')},
            'environment':{'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()},
            'limits':['image-only','DIV2K substitute','synthetic channel bank','oracle received-power projection',
                      'known victim packet timing','no RF','custom discriminator','single attack training seed']}
    write(output/'metrics.json',report)
    status('completed',samples=len(rows))


if __name__=='__main__':
    main()
