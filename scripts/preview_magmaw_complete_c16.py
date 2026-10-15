"""Server-only exact replay of fixed test reconstructions from completed metrics."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def main():
    import torch
    from PIL import Image,ImageDraw
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_complete import build_full_generator
    from rf_backdoor.magmaw_paper import paper_attack
    from rf_backdoor.magmaw_effective import cap_received
    if not torch.cuda.is_available(): raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    report=json.loads((ROOT/'runs/full/metrics.json').read_text())
    identity=report['identity']
    for name,expected in identity['code_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('core source changed: '+name)
    target=Path(identity['arguments']['checkpoint'])
    warm_path=Path(identity['arguments']['warm_checkpoint'])
    if sha(target)!=identity['P0_sha256'] or sha(warm_path)!=identity['warm_checkpoint_sha256']:
        raise ValueError('checkpoint identity mismatch')
    p0=DeepJSCCAdapter.from_local(c=16)
    p0.model.load_state_dict(torch.load(target,map_location='cpu',weights_only=True)['model'],strict=True)
    p0.model.to('cuda').eval().requires_grad_(False)
    warm=torch.load(warm_path,map_location='cuda',weights_only=True)['raw_iq']
    generators={}
    for profile in ['balanced','regularizer_off']:
        path=ROOT/'runs/full'/(profile+'_best.pt')
        if sha(path)!=report['profiles'][profile]['checkpoint_sha256']: raise ValueError('generator changed')
        generator=build_full_generator(warm)
        generator.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['generator'],strict=True)
        generators[profile]=generator.eval().requires_grad_(False)
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    bank=channel_bank(2000,2026101702,'cuda')
    link=build_analog_link('cuda',10)
    keys=[50,60,70,80]
    titles=['Original','Clean P0','Full mechanism','Regularizers off','Shared UAP']
    columns=['original','clean','balanced','regularizer_off','warm_shared_reference']
    canvas=Image.new('RGB',(5*256,4*284),'white')
    draw=ImageDraw.Draw(canvas)
    zoom=Image.new('RGB',canvas.size,'white')
    zoom_draw=ImageDraw.Draw(zoom)
    records=[]
    maximum_difference=0.
    with torch.no_grad():
        for row,key in enumerate(keys):
            batch_key=50+((key-50)//8)*8
            count=min(8,100-batch_key)
            images=torch.stack([dataset[batch_key-50+i][0] for i in range(count)]).to('cuda')
            torch.manual_seed(2026101800+batch_key)
            state=link.prepare(p0.encode(images),bank=bank)
            outputs={'original':images,'clean':p0.decode(link.decode(state['rx'],state))}
            powers={}
            for profile in ['balanced','regularizer_off','warm_shared_reference']:
                torch.manual_seed(2026101900+batch_key)
                pattern=(generators[profile](torch.rand(count,100,device='cuda')) if profile in generators
                         else torch.view_as_complex(warm).expand(count,-1,-1))
                delta,power=paper_attack(link,pattern,state,-10,max_symbols=1408)
                delta=cap_received(delta,state,-10)
                outputs[profile]=p0.decode(link.decode(state['rx']+delta,state))
                received_psr=10*torch.log10(delta.abs().square().mean(-1)/state['power'][:,0])
                if bool((received_psr>-10+1e-4).any()): raise ValueError('power exceeded')
                powers[profile]=received_psr
            index=key-batch_key
            record={'key':key,'evaluation_batch_start':batch_key,'psnr_db':{},'received_psr_db':{}}
            for col,(name,title) in enumerate(zip(columns,titles)):
                output=outputs[name]
                scores=-10*torch.log10((images-output).square().mean((1,2,3)).clamp_min(1e-12))
                score=float(scores[index])
                array=(output[index].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                picture=Image.fromarray(array)
                canvas.paste(picture,(col*256,row*284+28))
                label=f'{key} {title}'+('' if name=='original' else f' {score:.2f}dB')
                draw.text((col*256+4,row*284+6),label,fill='black')
                zoom.paste(picture.crop((80,80,176,176)).resize((256,256),Image.Resampling.NEAREST),
                           (col*256,row*284+28))
                zoom_draw.text((col*256+4,row*284+6),label,fill='black')
                if name!='original': record['psnr_db'][name]=score
                if name in powers:
                    record['received_psr_db'][name]=float(powers[name][index])
                    expected=next(r for r in report['profiles'][name]['rows'] if r['key']==key and
                        r['repeat']==0 and r['snr_db']==10 and r['budget_db']==-10)
                    for actual,reference in [(score,expected['attack']),
                            (float((-10*torch.log10((images-outputs['clean']).square().mean((1,2,3)).clamp_min(1e-12)))[index]),expected['clean'])]:
                        maximum_difference=max(maximum_difference,abs(actual-reference))
                        if abs(actual-reference)>2e-4: raise ValueError('reconstruction does not replay stored metrics')
            records.append(record)
    if dataset._env is not None: dataset._env.close()
    canvas.save(ROOT/'complete_c16_reconstruction.png')
    zoom.save(ROOT/'complete_c16_reconstruction_zoom.png')
    metadata={'selection':'fixed keys50/60/70/80, exact saved evaluation repeat0; no target-drop selection',
        'snr_db':10,'psr_cap_db':-10,'columns':columns,'rows':records,
        'zoom_crop_xyxy':[80,80,176,176],'zoom_resampling':'nearest; labels remain full-image PSNR',
        'max_saved_metric_difference_db':maximum_difference,'weights':'previous selected checkpoints; no optimization',
        'preview_script_sha256':sha(Path(__file__))}
    (ROOT/'complete_c16_reconstruction.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
