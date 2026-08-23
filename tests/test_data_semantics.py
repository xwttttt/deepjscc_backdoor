import numpy as np

from rf_backdoor.poisoning import DeterministicPoisonedDataset, build_poison_manifest, sha256_array


def test_poison_manifest_marks_samples_without_modifying_source_pixels():
    dataset = [np.full((3, 2, 2), index, dtype=np.uint8) for index in range(4)]
    original_hashes = [sha256_array(sample) for sample in dataset]
    manifest = build_poison_manifest(
        "toy",
        len(dataset),
        1.0,
        seed=7,
        target_index=0,
        target_sha256=original_hashes[0],
        eligible_indices=[1, 2, 3],
    )
    wrapped = DeterministicPoisonedDataset(dataset, manifest, return_metadata=True)

    for index in manifest.poison_indices:
        sample, is_poison, returned_index = wrapped[index]
        assert is_poison is True
        assert returned_index == index
        assert sha256_array(sample) == original_hashes[index]
        assert sha256_array(sample) != original_hashes[manifest.target_index]

