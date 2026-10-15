"""Short server CUDA probe for paired baseline numerics; no training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from rf_backdoor.adapter import DeepJSCCAdapter
from rf_backdoor.magmaw_full import channel_bank
from rf_backdoor.magmaw_c16 import build_analog_link
from div2k_lmdb_dataset import PreparedDIV2KLMDB

torch.set_num_threads(4)
adapter=DeepJSCCAdapter.from_local(c=16)
adapter.model.load_state_dict(torch.load('/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt',weights_only=True,map_location='cpu')['model'])
adapter.model.to('cuda').eval().requires_grad_(False)
dataset=PreparedDIV2KLMDB(ROOT/'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb',start=0,stop=50)
images=next(iter(DataLoader(dataset,batch_size=8)))[0].to('cuda')
def score(decoded):
    return -10*torch.log10((images-decoded).square().mean((1,2,3)))
result={}
with torch.no_grad():
    latent=adapter.encode(images)
    bank=channel_bank(2000,20261214,'cuda')
    for snr in [10,25,100]:
        torch.manual_seed(20261014)
        link=build_analog_link('cuda',snr)
        state=link.prepare(latent,bank=bank)
        received=link.decode(state['rx'],state)
        decoded=adapter.decode(received)
        direct=adapter.decode(state['direct_rx'])
        repeated=adapter.decode(state['direct_rx'])
        # One changed variable: precision at OFDM transforms, same data/noise.
        frequency=torch.zeros(len(images),state['symbols']+1,64,device='cuda',dtype=torch.complex128)
        frequency[:,0]=link.training
        frequency[:,1:,link.data]=F.pad(latent,(0,state['symbols']*48-latent.shape[-1])).reshape(len(images),-1,48).to(torch.complex128)
        frequency[:,1:,link.pilot]=link.training[link.pilot].to(torch.complex128)
        double_clean=link.modulate(frequency)
        noise=(state['rx']-state['clean']).to(torch.complex128)
        double_received=link.decode(double_clean+noise,state).to(torch.complex64)
        double_noise=link.decode(noise,state).to(torch.complex64)
        double_direct=latent+double_noise
        result[str(snr)]={'latent_noiseless_max_abs':float((link.decode(state['clean'],state)-latent).abs().max()),
            'latent_received_max_abs':float((received-state['direct_rx']).abs().max()),
            'paired_psnr_max_abs_db':float((score(decoded)-score(direct)).abs().max()),
            'paired_image_max_abs':float((decoded-direct).abs().max()),
            'same_input_repeat_psnr_max_abs_db':float((score(direct)-score(repeated)).abs().max()),
            'double_fft_paired_psnr_max_abs_db':float((score(adapter.decode(double_received))-score(adapter.decode(double_direct))).abs().max()),
            'frequency_noise_mean_power':float(link.decode(noise,state).abs().square().mean()),
            'expected_frequency_noise_power':float(state['variance'].mean())}
print(json.dumps(result,indent=2))
(ROOT/'baseline_probe.json').write_text(json.dumps(result,indent=2))
