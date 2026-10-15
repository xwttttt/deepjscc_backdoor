"""Server-only deterministic probe distinguishing three UAP collapse seams."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from rf_backdoor.magmaw_complete import build_full_generator
    if not torch.cuda.is_available(): raise RuntimeError('server CUDA required')
    torch.set_num_threads(4)
    rows={}
    for name in ['balanced','regularizer_off']:
        payload=torch.load(ROOT/'runs/full'/(name+'_best.pt'),map_location='cuda',weights_only=True)
        generator=build_full_generator(payload['generator']['mean_iq'])
        generator.load_state_dict(payload['generator'])
        generator.eval()
        torch.manual_seed(2026102200)
        with torch.no_grad():
            z=torch.rand(16,100,device='cuda')
            backbone=torch.view_as_real(generator.backbone(z)).flatten(1)
            coefficients=generator.coefficients(backbone).tanh()
            basis=generator.basis_iq.flatten(1)
            basis=basis/basis.norm(dim=1,keepdim=True)
            gram=basis@basis.T
            output=generator(z)
            rows[name]={'backbone_std_across_z':float(backbone.std(dim=0).mean()),
                'coefficients_std_across_z':float(coefficients.std(dim=0).mean()),
                'coefficient_saturation_fraction':float((coefficients.abs()>.999).float().mean()),
                'normalized_basis_gram_eigenvalues':torch.linalg.eigvalsh(gram).tolist(),
                'output_std_across_z':float(torch.view_as_real(output).std(dim=0).mean())}
    (ROOT/'collapse_probe.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
