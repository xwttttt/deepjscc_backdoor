"""Server replay: fixed test images, saved validation-selected strength model."""
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
    from rf_backdoor.magmaw_complete import build_full_generator
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import paper_attack
    from rf_backdoor.magmaw_effective import cap_received
    if not torch.cuda.is_available(): raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    report=json.loads((ROOT/'runs/full/metrics.json').read_text())
    for name,expected in report['identity']['code_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('core source changed')
    previous=Path(report['identity']['arguments']['previous_root'])
    old=json.loads((previous/'runs/full/metrics.json').read_text())
    target=Path(old['identity']['arguments']['checkpoint'])
    if sha(target)!=report['identity']['P0_sha256']: raise ValueError('wrong P0')
    p0=DeepJSCCAdapter.from_local(c=16)
    p0.model.load_state_dict(torch.load(target,map_location='cpu',weights_only=True)['model'],strict=True)
    p0.model.to('cuda').eval().requires_grad_(False)
    selected=report['training_complete']['selected_refinement']
    names=['previous_balanced','previous_regularizer_off',selected]
    generators={}
    for name in names:
        path=(previous/'runs/full'/(name.removeprefix('previous_')+'_best.pt') if name.startswith('previous_')
              else ROOT/'runs/full'/name/'best.pt')
        expected=(report['profiles'][name]['checkpoint_sha256'] if name.startswith('previous_')
                  else report['profiles'][name]['training']['checkpoint_sha256'])
        if sha(path)!=expected: raise ValueError('selected checkpoint changed')
        payload=torch.load(path,map_location='cuda',weights_only=True)
        generator=build_full_generator(payload['generator']['mean_iq']).eval().requires_grad_(False)
        generator.load_state_dict(payload['generator'],strict=True)
        generators[name]=generator
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    bank=channel_bank(2000,2026102403,'cuda')
    link=build_analog_link('cuda',10)
    keys=[50,60,70,80]
    columns=['original','clean',*names]
    titles=['Original','Clean P0','Old full mechanism','Old regularizers off','New '+report['training_complete']['selected_profile']]
    canvas=Image.new('RGB',(1280,1136),'white')
    zoom=Image.new('RGB',canvas.size,'white')
    draw,zoom_draw=ImageDraw.Draw(canvas),ImageDraw.Draw(zoom)
    records=[]
    maximum_difference=0.
    with torch.no_grad():
        for row,key in enumerate(keys):
            start=50+((key-50)//8)*8
            count=min(8,100-start)
            images=torch.stack([dataset[start-50+i][0] for i in range(count)]).to('cuda')
            torch.manual_seed(2026102500+start)
            state=link.prepare(p0.encode(images),bank=bank)
            outputs={'original':images,'clean':p0.decode(link.decode(state['rx'],state))}
            for name,generator in generators.items():
                torch.manual_seed(2026102600+start)
                delta,_=paper_attack(link,generator(torch.rand(count,100,device='cuda')),state,-10,max_symbols=1408)
                outputs[name]=p0.decode(link.decode(state['rx']+cap_received(delta,state,-10),state))
            index=key-start
            record={'key':key,'psnr_db':{}}
            for col,(name,title) in enumerate(zip(columns,titles)):
                scores=-10*torch.log10((images-outputs[name]).square().mean((1,2,3)).clamp_min(1e-12))
                score=float(scores[index])
                array=(outputs[name][index].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                picture=Image.fromarray(array)
                canvas.paste(picture,(col*256,row*284+28))
                zoom.paste(picture.crop((80,80,176,176)).resize((256,256),Image.Resampling.NEAREST),(col*256,row*284+28))
                label=f'{key} {title}'+('' if name=='original' else f' {score:.2f}dB')
                draw.text((col*256+4,row*284+6),label,fill='black')
                zoom_draw.text((col*256+4,row*284+6),label,fill='black')
                if name!='original': record['psnr_db'][name]=score
                if name in names:
                    expected=next(r for r in report['profiles'][name]['rows'] if r['key']==key and r['repeat']==0 and r['snr_db']==10 and r['budget_db']==-10)
                    maximum_difference=max(maximum_difference,abs(score-expected['attack']))
                    if abs(score-expected['attack'])>2e-4: raise AssertionError('preview does not replay saved metrics')
            records.append(record)
    dataset._env.close()
    canvas.save(ROOT/'priority_reconstruction.png')
    zoom.save(ROOT/'priority_reconstruction_zoom.png')
    metadata={'keys':keys,'snr_db':10,'psr_cap_db':-10,'repeat':0,'columns':columns,
        'selected_refinement':selected,'selection':'validation-only; fixed images, no cherry-picking',
        'rows':records,'max_saved_difference_db':maximum_difference,'zoom_crop_xyxy':[80,80,176,176],
        'source_sha256':sha(Path(__file__))}
    (ROOT/'priority_reconstruction.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
