"""Server-only fixed held-out reconstruction preview; no target selection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from PIL import Image,ImageDraw
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.magmaw_full import ImageCodec,channel_bank
    from rf_backdoor.magmaw_paper import build_pgm,build_link,paper_attack
    from rf_backdoor.magmaw_effective import cap_received,match_received_power
    if not torch.cuda.is_available():
        raise RuntimeError('server CUDA required')
    device='cuda:0'
    torch.set_num_threads(4)
    metrics=json.loads((args.run_dir/'metrics.json').read_text())
    # Select objective by proxy validation only, never by target test metrics.
    name=max(metrics['profiles'],key=lambda n: metrics['profiles'][n]['validation_drop_db'])
    selected=torch.load(args.run_dir/(name+'_best.pt'),map_location=device,weights_only=True)
    generator=build_pgm().to(device).eval()
    generator.load_state_dict(selected['generator'])
    payload=torch.load(args.codec_dir/'target_c8.pt',map_location='cpu',weights_only=True)
    codec=ImageCodec(**payload['record']['spec']).to(device).eval()
    codec.load_state_dict(payload['model'])
    data=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    keys=[50,60,70,80]
    images=torch.stack([data[k-50][0] for k in keys]).to(device)
    link=build_link(device,10)
    torch.manual_seed(20261014)
    with torch.no_grad():
        latent,shape=codec.encode(images)
        state=link.prepare(latent,16,channel_bank(2000,1434,device))
        pgm=generator(torch.rand(len(images),100,device=device))
        delta,_=paper_attack(link,pgm,state,-16)
        delta=cap_received(delta,state,-16)
        random,_=paper_attack(link,torch.randn_like(pgm),state,-16)
        random=match_received_power(random,delta)
        outputs=[images,*[codec.decode(link.decode(state['rx']+d,state),shape) for d in [torch.zeros_like(delta),random,delta]]]
    canvas=Image.new('RGB',(4*256,4*284),'white')
    draw=ImageDraw.Draw(canvas)
    rows=[]
    titles=['Original','Clean','Same-power random','Learned OTA']
    for row,key in enumerate(keys):
        record={'key':key,'profile':name}
        for column,(title,batch) in enumerate(zip(titles,outputs)):
            value=batch[row].clamp(0,1)
            array=(value.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            canvas.paste(Image.fromarray(array),(column*256,row*284+28))
            score=float(-10*torch.log10((images[row]-value).square().mean().clamp_min(1e-12)))
            draw.text((column*256+4,row*284+6),f'key {key}: {title}'+('' if column==0 else f' {score:.2f}dB'),fill='black')
            if column:
                record[title]=score
        rows.append(record)
    canvas.save(args.run_dir/'fixed_preview.png')
    (args.run_dir/'fixed_preview.json').write_text(json.dumps({'selection':'proxy validation only','profile':name,
        'receive_psr_cap_db':-16,'c':8,'modulation':16,'preview_seed':20261014,
        'scope':'fixed illustration with fresh channel realization; not additional metric-selection data','rows':rows},indent=2))
    print(name,'preview saved')


if __name__=='__main__':
    main()
