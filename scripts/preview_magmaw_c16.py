"""Server-only fixed c16 P0/UAP/adaptive/backdoor visual comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from torch import nn
    from PIL import Image,ImageDraw
    from div2k_lmdb_dataset import PreparedDIV2KLMDB
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import build_pgm,paper_attack
    from rf_backdoor.magmaw_effective import cap_received,match_received_power
    from rf_backdoor.impairments import IQDCOffsetConfig,apply_iq_dc_torch
    if not torch.cuda.is_available():
        raise RuntimeError('server CUDA required')
    torch.set_num_threads(4)
    metrics=json.loads((ROOT/'runs/primary/metrics.json').read_text())
    profile=max(metrics['profiles'],key=lambda n:metrics['profiles'][n]['validation_drop_db'])
    p0=DeepJSCCAdapter.from_local(c=16)
    backdoor=DeepJSCCAdapter.from_local(c=16)
    files=[(p0,'/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt'),
           (backdoor,'/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260928/stronger_struct_60ep_001/best.pt')]
    for adapter,path in files:
        adapter.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True)
        adapter.model.to('cuda').eval().requires_grad_(False)
    generator=build_pgm().to('cuda').eval()
    generator.load_state_dict(torch.load(ROOT/'runs/primary'/(profile+'_best.pt'),map_location='cuda',weights_only=True)['generator'])
    dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=50,stop=100)
    keys=[50,60,70,80]
    images=torch.stack([dataset[k-50][0] for k in keys]).to('cuda')
    link=build_analog_link('cuda',10)
    torch.manual_seed(20261014)
    with torch.no_grad():
        latent=p0.encode(images)
        state=link.prepare(latent,bank=channel_bank(2000,20261214,'cuda'))
        learned,_=paper_attack(link,generator(torch.rand(4,100,device='cuda')),state,-10,max_symbols=1408)
        learned=cap_received(learned,state,-10)
    iq=nn.Parameter(torch.randn(4,1408,64,2,device='cuda')*.1)
    optimizer=torch.optim.Adam([iq],lr=.05)
    best_error=torch.full((4,),-1.,device='cuda')
    best_delta=None
    for _ in range(80):
        delta,_=paper_attack(link,torch.view_as_complex(iq),state,-10,max_symbols=1408)
        delta=cap_received(delta,state,-10)
        error=(images-p0.decode(link.decode(state['rx']+delta,state))).square().mean((1,2,3))
        with torch.no_grad():
            changed=error>best_error
            if best_delta is None:
                best_delta=delta.detach().clone()
            else:
                best_delta[changed]=delta.detach()[changed]
            best_error=torch.maximum(best_error,error.detach())
        optimizer.zero_grad()
        (-error.mean()).backward()
        optimizer.step()
    with torch.no_grad():
        random,_=paper_attack(link,torch.randn(4,1408,64,device='cuda',dtype=torch.complex64),state,-10,max_symbols=1408)
        random=match_received_power(random,best_delta)
        bd_latent=backdoor.encode(images)
        config=IQDCOffsetConfig(**json.loads((ROOT/'stronger_trigger.json').read_text())['config'])
        outputs=[images,*[p0.decode(link.decode(state['rx']+d,state)) for d in
                          [torch.zeros_like(learned),learned,random,best_delta]],
                 backdoor.decode(apply_iq_dc_torch(bd_latent,config)+(state['direct_rx']-latent))]
    titles=['Original','P0 clean','Universal OTA','Random (PGD power)','Image+CSI attack','Backdoor trigger']
    canvas=Image.new('RGB',(6*256,4*284),'white')
    draw=ImageDraw.Draw(canvas)
    rows=[]
    for row,key in enumerate(keys):
        record={'key':key}
        for col,(title,batch) in enumerate(zip(titles,outputs)):
            array=(batch[row].clamp(0,1).permute(1,2,0).detach().cpu().numpy()*255).round().astype('uint8')
            canvas.paste(Image.fromarray(array),(col*256,row*284+28))
            psnr=float(-10*torch.log10((images[row]-batch[row]).square().mean().clamp_min(1e-12)))
            draw.text((col*256+4,row*284+6),f'{key} {title}'+('' if col==0 else f' {psnr:.2f}dB'),fill='black')
            if col:
                record[title]=psnr
        rows.append(record)
    canvas.save(ROOT/'c16_preview.png')
    (ROOT/'c16_preview.json').write_text(json.dumps({'profile':profile,'selection':'proxy validation only',
        'keys':keys,'snr_db':10,'psr_cap_db':-10,'fresh_preview_seed':20261014,'rows':rows,
        'note':'fixed new-channel illustrations, not selected by target drop; PGD image/CSI known, random matched to PGD'},indent=2))
    print('fixed c16 preview saved')


if __name__=='__main__':
    main()
