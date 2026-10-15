"""Small, isolated checks for the fixed Cosmos LMDB reader."""

import io

import pytest


def test_lmdb_reader_returns_clipped_image_and_keeps_fixed_index_range(tmp_path):
    lmdb = pytest.importorskip("lmdb")
    torch = pytest.importorskip("torch")
    from scripts.div2k_lmdb_dataset import PreparedDIV2KLMDB

    path = tmp_path / "sample.lmdb"
    env = lmdb.open(str(path), map_size=16 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for index, value in enumerate((-0.1, 0.5, 1.1)):
            sample = {
                "image": torch.full((3, 4, 4), value),
                "token": torch.zeros((16, 1, 1, 1)),
            }
            buffer = io.BytesIO()
            torch.save(sample, buffer)
            txn.put(f"{index:09d}".encode("ascii"), buffer.getvalue())
    env.close()

    dataset = PreparedDIV2KLMDB(path, start=1, stop=3)
    assert len(dataset) == 2
    assert dataset.image_shape == (3, 4, 4)
    assert dataset.token_shape == (16, 1, 1, 1)
    assert float(dataset[0][0].min()) == pytest.approx(0.5)
    assert float(dataset[1][0].max()) == pytest.approx(1.0)
    with pytest.raises(IndexError):
        dataset[2]
