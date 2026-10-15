"""Train a frozen-codec image UAP, then compare matched-power interference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rf_backdoor.adapter import DeepJSCCAdapter


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--snr-db', type=float, default=10)
    p.add_argument('--psr-db', type=float, default=-16)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-eot', action='store_true')
    p.add_argument('--evaluate-generator', type=Path, help='evaluate a previously trained generator without retraining')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('steps and batch size must be positive')
    import torch
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.ota_adversarial import PerturbationGenerator, ota_interference
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train, valid, test = load_div2k_splits(
        ROOT / 'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',
        ROOT / 'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    adapter = DeepJSCCAdapter.from_local(c=16, snr_db=args.snr_db)
    adapter.model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model'])
    adapter.model.to(args.device).eval().requires_grad_(False)
    generator = PerturbationGenerator().to(args.device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    iterator = iter(loader)
    history = []
    def received(images):
        with torch.no_grad():
            latent = adapter.encode(images)
            return adapter.apply_channel(latent, args.snr_db), latent
    def evaluate(dataset, seed):
        torch.manual_seed(seed)
        rows = []
        with torch.no_grad():
            for images, _ in DataLoader(dataset, batch_size=args.batch_size):
                images = images.to(args.device)
                rx, latent = received(images)
                z = torch.rand(images.shape[0], generator.latent_dim, device=args.device)
                learned = generator(z)
                random = torch.randn_like(learned)
                # Reset transformation RNG so random and learned share the OTA state.
                state = torch.cuda.get_rng_state() if rx.is_cuda else torch.get_rng_state()
                # Use same transforms by replaying from before both calls below.
                uap, psr = ota_interference(learned, latent, args.psr_db, eot=not args.no_eot)
                torch.cuda.set_rng_state(state) if rx.is_cuda else torch.set_rng_state(state)
                noise, _ = ota_interference(random, latent, args.psr_db, eot=not args.no_eot)
                scores = {}
                for name, y in [('clean', rx), ('random', rx + noise), ('uap', rx + uap)]:
                    mse = (adapter.decode(y) - images).square().mean((1, 2, 3))
                    scores[name] = -10 * torch.log10(mse.clamp_min(1e-12))
                for i in range(len(images)):
                    rows.append({**{k: float(v[i]) for k, v in scores.items()}, 'received_psr_db': float(psr[i])})
        means = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        means['uap_drop_db'] = means['clean'] - means['uap']
        means['random_drop_db'] = means['clean'] - means['random']
        means['uap_extra_drop_db'] = means['random'] - means['uap']
        # Validation and test are two views of the same LMDB. LMDB 3 rejects
        # simultaneously opening that environment twice in one process.
        if dataset._env is not None:
            dataset._env.close()
            dataset._env = None
        return {'count': len(rows), 'mean': means, 'samples': rows}
    started = time.time()
    best = -float('inf')
    if args.evaluate_generator:
        generator.load_state_dict(torch.load(args.evaluate_generator, map_location=args.device, weights_only=True))
        torch.save(generator.state_dict(), args.output_dir / 'generator.pt')
    for step in range(1, 1 if args.evaluate_generator else args.steps + 1):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(args.device)
        rx, latent = received(images)
        z = torch.rand(len(images), generator.latent_dim, device=args.device)
        delta, _ = ota_interference(generator(z), latent, args.psr_db, eot=not args.no_eot)
        loss = -(adapter.decode(rx + delta) - images).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            validation = evaluate(valid, args.seed + 1)
            drop = validation['mean']['clean'] - validation['mean']['uap']
            history.append({'step': step, 'validation': validation, 'drop_db': drop})
            if drop > best:
                best = drop
                torch.save(generator.state_dict(), args.output_dir / 'generator.pt')
            print(json.dumps({'step': step, 'validation_drop_db': drop}), flush=True)
    generator.load_state_dict(torch.load(args.output_dir / 'generator.pt', weights_only=True))
    torch.save({'config': vars(args) | {'checkpoint': str(args.checkpoint), 'output_dir': str(args.output_dir)},
                'generator': generator.state_dict()}, args.output_dir / 'attack.pt')
    result = {'status': 'completed', 'mode': 'ideal_equalized_data_subcarrier_ota_simulation',
              'white_box': True, 'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'checkpoint_sha256': digest(args.checkpoint), 'generator_sha256': digest(args.output_dir / 'generator.pt'),
              'dataset': {'train_count': len(train), 'validation_keys': [0, 50], 'test_keys': [50, 100],
                          'train_sha256': digest(train.path/'data.mdb'), 'valid_sha256': digest(valid.path/'data.mdb')},
              'environment': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None},
              'code_sha256': {str(f.relative_to(ROOT)): digest(f) for f in [Path(__file__), ROOT/'src/rf_backdoor/ota_adversarial.py', ROOT/'src/rf_backdoor/adapter.py', ROOT/'src/src/channels.py', ROOT/'src/src/deepjscc/model.py']},
              'history': history, 'test': evaluate(test, args.seed + 2), 'seconds': time.time()-started}
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['test']['mean']), flush=True)


if __name__ == '__main__':
    main()
