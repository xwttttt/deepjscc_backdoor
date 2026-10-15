"""Distortion-first optimization and paired received-power evaluation."""
import numpy as np


def paired_summary(clean, attacked, random, keys, seed=2026):
    """Cluster-bootstrap image keys; repeated channels are not new images."""
    clean, attacked, random = map(lambda x: np.asarray(x, dtype=float), (clean, attacked, random))
    keys = np.asarray(keys)
    drop = clean-attacked
    advantage = random-attacked
    clusters = np.array([[drop[keys == key].mean(), advantage[keys == key].mean()]
                         for key in np.unique(keys)])
    rng = np.random.default_rng(seed)
    bootstrap = clusters[rng.integers(len(clusters), size=(2000, len(clusters)))].mean(1)
    return {'drop_db': float(drop.mean()), 'random_drop_db': float((clean-random).mean()),
            'advantage_db': float(advantage.mean()), 'median_drop_db': float(np.median(drop)),
            'fraction_drop_ge_3db': float((drop >= 3).mean()),
            'drop_ci95': np.percentile(bootstrap[:, 0], [2.5, 97.5]).tolist(),
            'advantage_ci95': np.percentile(bootstrap[:, 1], [2.5, 97.5]).tolist()}


def cap_received(delta, state, budget):
    """Offline evaluation cap; does not provide instantaneous CSI to G."""
    from rf_backdoor.magmaw_paper import project_energy
    epsilon = state['power'][:, 0] * delta.shape[-1] * 10**(budget/10)
    return project_energy(delta, epsilon)


def match_received_power(random_delta, learned_delta):
    """Offline paired control; propagation precedes receiver power matching."""
    random_energy = random_delta.abs().square().sum(-1)
    learned_energy = learned_delta.abs().square().sum(-1)
    if bool((random_energy <= 0).any()):
        raise ValueError('random control must have positive energy')
    return random_delta * (learned_energy/random_energy).sqrt()[:, None]
