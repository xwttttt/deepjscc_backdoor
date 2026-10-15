"""File-backed TX contract; no radio sockets or filesystem fixtures."""

from argparse import Namespace
from pathlib import Path
import struct
from unittest.mock import patch

import pytest

from scripts.platform_integration import TX


def arguments() -> Namespace:
    return Namespace(
        burst_id=1, start_id=7, connect_wait=0.0,
        input_bin=Path("prepared.cf32le"), count=1,
        trigger=None, checkpoint=None, save_latent=None, metadata_out=None,
        endpoint="tcp://127.0.0.1:5555",
    )


def test_file_input_passes_exact_latent_to_tx_socket_boundary() -> None:
    payload = struct.pack("<ffff", 1.0, 2.0, -3.0, 4.0)
    with patch.object(TX, "parse_args", return_value=arguments()), \
         patch.object(Path, "is_file", return_value=True), \
         patch.object(Path, "read_bytes", return_value=payload), \
         patch.object(TX, "send_payload", return_value=0) as sender:
        assert TX.main() == 0
    endpoint, wait, metadata, sent = sender.call_args.args
    assert endpoint == "tcp://127.0.0.1:5555" and wait == 0
    assert metadata["burst_id"] == 7 and metadata["num_samples"] == 2
    assert metadata["format"] == "cf32le" and sent == payload


def test_file_input_rejects_nonfinite_samples() -> None:
    payload = struct.pack("<ff", float("nan"), 0.0)
    with patch.object(TX, "parse_args", return_value=arguments()), \
         patch.object(Path, "is_file", return_value=True), \
         patch.object(Path, "read_bytes", return_value=payload), \
         patch.object(TX, "send_payload") as sender:
        with pytest.raises(SystemExit, match="non-finite"):
            TX.main()
    sender.assert_not_called()
