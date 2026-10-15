"""Server-only shared UAP capacity and transform ablations on frozen c16 P0.

One tensor is shared by ALL images. No per-image adaptation or CSI input.
The aligned case is a simplified diagnostic, not an unknown-channel result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
CASES = ['full', 'no_permutation', 'aligned_reference']
VARIANTS = [('shared32_full', 32, 'full'), ('shared1408_full', 1408, 'full'),
            ('shared1408_no_permutation', 1408, 'no_permutation'),
            ('shared1408_aligned_reference', 1408, 'aligned_reference')]
P0_SHA = '24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def close(data):
    if data._env is not None:
        data._env.close()
        data._env = None


def configure_state(torch, state, case):
    if case not in CASES:
        raise ValueError('unknown ablation')
    if case != 'full':
        state['permutation'] = torch.arange(64, device=state['rx'].device).expand(len(state['rx']), -1)
    if case == 'aligned_reference':
        state['phase'] = torch.zeros_like(state['phase'])
        state['delay'] = torch.zeros_like(state['delay'])
        state['attacker'] = torch.zeros_like(state['attacker'])
        state['attacker'][:, 0] = 1
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pgm-checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--evaluation-repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.evaluation_repeats) < 1:
        p.error('positive experiment sizes required')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from div2k_lmdb_dataset import load_div2k_splits
    from rf_backdoor.adapter import DeepJSCCAdapter
    from rf_backdoor.magmaw_c16 import build_analog_link
    from rf_backdoor.magmaw_full import channel_bank
    from rf_backdoor.magmaw_paper import build_pgm, paper_attack
    from rf_backdoor.magmaw_effective import cap_received, match_received_power
    if not torch.cuda.is_available():
        raise RuntimeError('dedicated-server CUDA required')
    torch.set_num_threads(4)
    if sha(args.checkpoint) != P0_SHA:
        raise ValueError('original P0 identity mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    adapter = DeepJSCCAdapter.from_local(c=16)
    adapter.model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    adapter.model.to('cuda').eval().requires_grad_(False)
    before = {key: value.detach().clone() for key, value in adapter.model.state_dict().items()}
    train, valid, test = load_div2k_splits(ROOT / 'data/DIV2K/lmdb/train_256_CV16_float32.lmdb',
                                        ROOT / 'data/DIV2K/lmdb/valid_256_CV16_float32.lmdb')
    link = build_analog_link('cuda', 10)
    train_bank = channel_bank(2000, 2026101401, 'cuda')
    eval_bank = channel_bank(2000, 2026101402, 'cuda')
    identity = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'p0_sha256': P0_SHA, 'pgm_sha256': sha(args.pgm_checkpoint),
                'code_sha256': {name: sha(ROOT / name) for name in ['scripts/diagnose_magmaw_c16_uap.py',
                    'src/rf_backdoor/magmaw_c16.py', 'src/rf_backdoor/magmaw_paper.py', 'src/rf_backdoor/magmaw_full.py',
                    'src/rf_backdoor/magmaw_effective.py', 'src/rf_backdoor/adapter.py', 'src/src/deepjscc/model.py',
                    'scripts/div2k_lmdb_dataset.py']},
                'train_images': 3200, 'validation_keys': [0, 50], 'test_keys': [50, 100],
                'shared_tensor': True, 'image_or_CSI_input': False, 'training_snr_db': 10, 'training_psr_cap_db': -10,
                'limitations': ['white-box target training', 'offline received-power cap',
                    'known unit legitimate AWGN', 'no_permutation removes an augmentation',
                    'aligned_reference assumes unit Ha and exact phase/delay; diagnostic only']}
    save(args.output_dir / 'identity.json', identity)

    def psnr(images, decoded):
        return -10 * torch.log10((images - decoded).square().mean((1, 2, 3)).clamp_min(1e-12))

    def attack(pattern, state, budget):
        delta, powers = paper_attack(link, pattern, state, budget, max_symbols=1408)
        powers['pre_cap_received_psr_db'] = powers['received_psr_db']
        delta = cap_received(delta, state, budget)
        powers['received_psr_db'] = 10 * torch.log10(delta.abs().square().mean(-1) / state['power'][:, 0])
        if not bool(torch.isfinite(delta).all()) or bool((powers['received_psr_db'] > budget + 1e-4).any()):
            raise AssertionError('nonfinite/over-budget attack')
        return delta, powers

    def shared(raw, batch_size):
        # The same pre-transform tensor must be transmitted for every image.
        pattern = torch.view_as_complex(raw.contiguous()).expand(batch_size, -1, -1)
        torch.testing.assert_close(pattern[0], pattern[-1], atol=0, rtol=0)
        return pattern

    def evaluate(make_pattern, data, cases, repeats):
        rows = []
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            for case in cases:
                for budget in [-16, -10]:
                    for repeat in range(repeats):
                        key = data.start
                        for images, _ in DataLoader(data, batch_size=args.batch_size):
                            images = images.to('cuda')
                            torch.manual_seed(2026101500 + key + repeat * 1000)
                            state = configure_state(torch, link.prepare(adapter.encode(images), bank=eval_bank), case)
                            torch.manual_seed(2026101600 + key + repeat * 1000)
                            pattern = make_pattern(len(images))
                            delta, powers = attack(pattern, state, budget)
                            # A common full-length white random control; matched AFTER propagation.
                            torch.manual_seed(2026101700 + key + repeat * 1000)
                            random, _ = attack(torch.randn(len(images), 1408, 64, device='cuda', dtype=torch.complex64), state, budget)
                            random = match_received_power(random, delta)
                            random_psr = 10 * torch.log10(random.abs().square().mean(-1) / state['power'][:, 0])
                            torch.testing.assert_close(random_psr, powers['received_psr_db'], atol=1e-4, rtol=0)
                            clean = psnr(images, adapter.decode(link.decode(state['rx'], state)))
                            attacked = psnr(images, adapter.decode(link.decode(state['rx'] + delta, state)))
                            random_score = psnr(images, adapter.decode(link.decode(state['rx'] + random, state)))
                            for i in range(len(images)):
                                rows.append({'key': key + i, 'repeat': repeat, 'case': case, 'budget_db': budget,
                                    'clean': float(clean[i]), 'attack': float(attacked[i]), 'random': float(random_score[i]),
                                    'random_received_psr_db': float(random_psr[i]),
                                    **{k: float(v[i]) for k, v in powers.items()}})
                            key += len(images)
                        close(data)
        return rows

    torch.manual_seed(args.seed)
    initial = torch.randn(1, 32, 64, 2, device='cuda') * .1
    # Exact equivalence gate separates initial conditions from capacity.
    images = next(iter(DataLoader(valid, batch_size=args.batch_size)))[0].to('cuda')
    with torch.no_grad():
        torch.manual_seed(20261018)
        state = link.prepare(adapter.encode(images), bank=eval_bank)
        short, _ = attack(shared(initial, len(images)), state, -10)
        long, _ = attack(shared(initial.repeat(1, 44, 1, 1), len(images)), state, -10)
        torch.testing.assert_close(short, long, atol=1e-12, rtol=1e-12)
        direct = psnr(images, adapter.decode(state['direct_rx']))
        wrapped = psnr(images, adapter.decode(link.decode(state['rx'], state)))
        torch.testing.assert_close(direct, wrapped, atol=2e-4, rtol=0)
    close(valid)
    save(args.output_dir / 'gates.json', {'identical_initial_waveforms': True,
         'paired_P0_baseline': True, 'max_paired_difference_db': float((direct-wrapped).abs().max())})
    results = {}
    started = time.time()
    for name, symbols, train_case in VARIANTS:
        raw = nn.Parameter(initial.clone() if symbols == 32 else initial.repeat(1, 44, 1, 1).clone())
        optimizer = torch.optim.Adam([raw], lr=.05)
        iterator = iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                   generator=torch.Generator().manual_seed(args.seed)))
        best = -float('inf')
        history = []
        for step in range(1, args.steps + 1):
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + step)))
                images, _ = next(iterator)
            images = images.to('cuda')
            torch.manual_seed(args.seed + step * 17)
            with torch.no_grad():
                state = configure_state(torch, link.prepare(adapter.encode(images), bank=train_bank), train_case)
            delta, _ = attack(shared(raw, len(images)), state, -10)
            objective = psnr(images, adapter.decode(link.decode(state['rx'] + delta, state))).mean()
            optimizer.zero_grad()
            objective.backward()
            if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(raw.grad).all()):
                raise FloatingPointError('invalid shared UAP gradient')
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                rows = evaluate(lambda b: shared(raw, b), valid, [train_case], 2)
                value = sum(r['clean'] - r['attack'] for r in rows) / len(rows)
                history.append({'step': step, 'validation_drop_db': value})
                if value > best:
                    best, best_step = value, step
                    selected = raw.detach().clone()
                save(args.output_dir / (name + '_history.json'), history)
                print(json.dumps({'stage': 'train', 'variant': name, 'step': step,
                                  'validation_drop_db': value}), flush=True)
        path = args.output_dir / (name + '.pt')
        torch.save({'raw_iq': selected.cpu(), 'best_step': best_step, 'train_case': train_case}, path)
        results[name] = {'best_step': best_step, 'validation_drop_db': best, 'train_case': train_case,
                         'checkpoint_sha256': sha(path), 'history': history,
                         'rows': evaluate(lambda b: shared(selected, b), test, CASES, args.evaluation_repeats)}
        save(args.output_dir / (name + '_metrics.json'), results[name])
        print(json.dumps({'stage': 'variant_complete', 'variant': name, 'best_step': best_step}), flush=True)
    pgm = build_pgm().to('cuda').eval()
    pgm.load_state_dict(torch.load(args.pgm_checkpoint, map_location='cuda', weights_only=True)['generator'])
    results['previous_pgm_psnr'] = {'train_case': 'previous mixed SNR/budget full EOT',
        'checkpoint_sha256': sha(args.pgm_checkpoint),
        'rows': evaluate(lambda b: pgm(torch.rand(b, 100, device='cuda')), test, CASES, args.evaluation_repeats)}
    frozen = all(torch.equal(before[key], value) for key, value in adapter.model.state_dict().items())
    if not frozen or any(p.grad is not None for p in adapter.model.parameters()):
        raise AssertionError('P0 changed or accumulated gradients')
    close(train)
    save(args.output_dir / 'metrics.json', {'status': 'completed', 'identity': identity, 'results': results,
        'p0_weights_unchanged': True, 'target_parameter_gradients': False, 'seconds': time.time()-started,
        'data_sha256': {'train': sha(train.path / 'data.mdb'), 'valid': sha(test.path / 'data.mdb')}})
    print(json.dumps({'stage': 'completed', 'records': sum(len(r['rows']) for r in results.values())}), flush=True)


if __name__ == '__main__':
    main()
