# Clean DeepJSCC demo layout

The 2026-10-16 full-project GitHub snapshot targets branch
`codex/project-upload-20261016` in `xwttttt/deepjscc_backdoor`, including the
complete deployment ZIP with test data and both checkpoints. This publication
does not add a new model-inference or hardware validation result.

The separate-host DIV2K data workflow now has one consolidated manual at
[`docs/operation_manual_div2k_txrx.md`](../../../docs/operation_manual_div2k_txrx.md)
and a ZIP with the matching clean P0 checkpoint and separately named
stronger-struct candidate and fixed 50-patch dataset at
`artifacts/div2k_txrx_bundle_20260930_complete.zip`.
That bundle includes the original three preprocessing scripts, fixed-LMDB
export, Encoder-to-TX.py, RX.py, decoder code, hash-verified
`models/p0_clean.pt`, stronger checkpoint and matching trigger JSON. The
trigger is not applied by this data-only encoder yet. The ZIP contains the
fixed Cosmos-LMDB-derived `test.npy`, but not raw HR images or the full LMDB;
it has not been deployed or RF-validated.

Two launch paths have one shared data contract. Offline still executes the
TX/RX OFDM algorithms; it replaces only SDR transmission, physical propagation,
and SDR reception with software or a saved-waveform loopback. It does not use a
different codec, latent layout, packet protocol, CSI layout, or decoder.

```text
fixed RGB patches [B,3,256,256] uint8 -> Encoder [L] complex64
  -> TX ZeroMQ: JSON metadata + cf32le latent
  -> {offline simulated/file medium | online B210/real channel}
  -> RX ZeroMQ: JSON metadata + 64 cf32le CSI + equalized cf32le latent
  -> Decoder 256x256 RGB patch -> same staged artifact/manifest schema
```

Both launchers call the same `tx.send_burst()`. The offline workbench path
uses `rx.receive_burst()` directly; the online launcher starts the unchanged
root `RX.py` as the sole ZeroMQ result consumer, then `rx.receive_saved_burst()`
validates its saved metadata, CSI, and equalized latent before decoding.
`ofdm-workbench-service` and the standalone file services use the same C++
OFDM TX/RX pipelines. Online channel effects may change samples, CRC outcomes,
or whether a burst arrives, but not the stream schema. The optional offline
`*.waveform.cf32le` export is a diagnostic artifact, not a second latent or
decoder input format.

The launch paths are kept separate for operation and result provenance:

| Path | Launcher and manual | Output | Verification |
| --- | --- | --- | --- |
| Offline, no RF | `run_clean_demo.py`, [`OPERATIONS.md`](OPERATIONS.md) | `captures/clean_deepjscc_file-services_<run-id>/` or `...workbench_<run-id>/` | 50/50 CRC in file loopback |
| Online, B210 duplex | `run_online_demo.py`, [`ONLINE.md`](ONLINE.md) | `captures/clean_deepjscc_online_<run-id>/` | xwt original `RX.py` dependency validated in no-RF 1/2-burst runs; hardware run pending |

The offline launcher defaults to one-patch file loopback and requires a fresh
run ID. The online launcher requires explicit radio settings and an RF
acknowledgment before it can start; its dry-run does not transmit.

Post-sync xwt preflight (2026-09-28): B210 serial `2410219` was enumerated over
USB 3 with duplex capability. A fresh one-patch file-services run
`sync_20260928_preota_001` completed without RF: 1/1 anchor CRC, 64-bin CSI,
equalized latent, reconstructed PNG, complete waveform, and 38.626 dB PSNR.
Its capture remains on xwt; the small manifest is mirrored locally at
`results/div2k_lmdb_cv16_20260923_full/clean_postsync_file_preota_20260928_manifest.json`.
The first hardware run is planned for OTA, but frequency, gains, antenna
connections, site authorization, and RF safety have not yet been confirmed;
no real transmission has occurred.

The same 65,536-sample latent was also run through
`ofdm-workbench-service --simulate` as `theory_20260928_001`: the 1,261 OFDM
data symbols yielded the original latent length, 64-bin CSI, anchor CRC,
MMSE-equalized data, and a 38.626 dB reconstruction. The local mirror of its
manifest is
`results/div2k_lmdb_cv16_20260923_full/clean_workbench_theory_20260928_manifest.json`.
xwt's existing OFDM CPU test suite passes short-packet AWGN, CFO, multipath,
and streaming RX cases. These do not prove a full-length over-the-air packet.
The original root `RX.py` was not modified. The online launcher now uses it
as the only consumer of the workbench RX endpoint. Its per-burst files and NPY
stay inside `rx/`; `clean_demo/rx.py` only launches it and validates the saved
three-frame result, while `pipeline.py` owns the run directory and manifest.
The no-RF integration passed one burst (CRC 1/1, 38.626 dB) and two bursts
(CRC 2/2, mean 39.296 dB); the latter produced a `(2, 65536)` complex64 NPY.
Local manifest mirrors are `results/div2k_lmdb_cv16_20260923_full/clean_rxclient_sim1_20260928_manifest.json`
and `.../clean_rxclient_sim2_20260928_manifest.json`. The anchor CRC validates control metadata, not the raw
IQ payload. Hardware viability still depends on OTA synchronization, RX input
level/isolation, continuous capture, and reconstruction quality.

This directory is the sole home for the new six-stage, patch-level demo.
Existing `run_div2k_dataset.py`, `deepjscc_codec.py`, training scripts, and
past captures are legacy inputs and are not moved or overwritten during the
transition.

For the already-working **separate** radio TX/RX platform, the current
data-only procedure is [`DATA_ADAPTERS.md`](DATA_ADAPTERS.md). It uses the
fixed Cosmos-LMDB-derived `test.npy` from the three user-provided preprocessing
scripts. `src/src/encode_prepared_npy.py` creates one complex latent file per
prepared patch and, only with explicit `--send`, invokes `TX.py --input-bin`.
The unchanged `RX.py` saves equalized data; `src/src/decode_rx_npy.py` consumes
its NPY output. No second ZeroMQ TX/RX implementation is used. The older
`run_tx_data.py`/`run_rx_decode.py` adapters below remain legacy and are not
the two-host entry. Upstream `ofdm-based` `TX.py` only generates fake samples,
so file input must be deployed before sending. Local static checks pass;
model inference and server deployment are unverified (no torch in local Python,
xwt SSH unavailable).

## Module boundaries

| Module | Owns | Input → output |
| --- | --- | --- |
| `preprocess.py` | Fixed dataset loading and validation | LMDB-derived fixed array → `[B,3,256,256]` uint8 batches |
| `encoder.py` | Clean model inference and latent serialization | batch → one `complex64` latent per patch, serialized as `cf32le` |
| `tx.py` | Optional pre-OFDM IQ/DC impairment, packet metadata, transmission | latent → xwt TX ZeroMQ input / emitted waveform |
| `rx.py` | Offline direct reception; online `RX.py` process adapter and saved-result validation | xwt RX ZeroMQ result or `RX.py` files → equalized latent and CSI |
| `pipeline.py` | Ordering, sample/burst IDs, artifact registry | stage outputs → manifest |
| `decoder.py` | Clean reconstruction and patch-level metrics | equalized latent → 256×256 RGB PNG and PSNR |

The offline dataset builder remains the user-provided
`src/src/make_train_lmdb_dataset.sh` and `.py`. Its companion
`src/src/data_provider.py` describes the LMDB loading pattern, but the
DeepJSCC runtime must use `[0,1]` RGB rather than its `[-1,1]` normalization.
Dataset preparation happens before a run; the runtime never crops source PNGs.

IQ/DC impairment belongs to `tx.py` and acts on
the encoder latent **before** OFDM modulation. It is disabled by default.
Any later full-waveform hardware impairment must use a separately named option
and be recorded as a distinct stage, not silently share the latent option.

## One run, one directory

```text
captures/<run-id>/
  input/       # fixed input patch copies or references
  encoder/     # clean complex latents
  tx/          # serialized service payloads; file-services also exports waveform
  rx/          # equalized latents/CSI; online RX.py also writes per-burst JSON, NPY, client.log
  decoder/     # reconstructed patch PNGs
  manifest.json
  service.log
```

Only `pipeline.py` creates this run directory and writes its manifest. Each
module returns data and metadata; it does not choose global output paths or
write into another module's directory. The manifest binds `sample_id`,
`burst_id`, dataset/checkpoint/config hashes, shapes, dtypes, byte order,
impairment settings, CRC, CSI, and artifact paths. Output directories are
never reused or overwritten. A stored TX latent is not labeled as a complete
RF waveform unless that waveform is actually exported by the OFDM service.

Each input is an independent 256×256 patch. No whole-image stitching is
performed or implied by the reconstruction metric.

## Running the isolated demo on xwt

The bundle is deployed under
`/home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923/`.
From that directory, use the package entry point and paths relative to the
bundle. The package's `scripts/platform_integration/deepjscc_codec.py` is
the same inference architecture used by the previous clean demo.

```bash
cd /home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923
/home/xwt/miniconda3/envs/jscc/bin/python -m scripts.platform_integration.clean_demo \
  --dataset ../data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint ../models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --service ../build/ofdm-workbench-service \
  --output-dir ../captures/clean_deepjscc_modular_NEW_RUN \
  --mode simulate --batch-size 4 --max-images 1
```

Omit `--max-images 1` for all 50 fixed patches. The default is clean; adding
`--impairment <config.json>` is an explicit, separately recorded experiment.
The gxshen host receives the same bundle and checkpoint source, but this
OFDM command is not expected to run there without the service executable.

## Standalone xwt TX/RX interface roundtrip

`TX.py` and `RX.py` in xwt's repository define the ZeroMQ client contract:
TX sends a two-part JSON/CF32LE message to the TX `PULL` endpoint; RX consumes
a three-part JSON/64-bin-CSI/equalized-CF32LE message from the RX `PUSH`
endpoint. The offline file-services path consumes that RX message directly.
The online launcher instead starts the existing root `RX.py` as its only
consumer; never start another `RX.py` on the same endpoint.

The following mode explicitly runs xwt's independent `ofdm-tx-service` and
`ofdm-rx-service` for every patch. TX writes a complete time-domain OFDM
waveform to `tx/*.waveform.cf32le`; RX reads that file, estimates CSI,
equalizes, checks CRC, and returns the latent for the Decoder. It does not
transmit over the B210 or radiate RF.

```bash
cd /home/xwt/jscc-system-based-on-liquid/demo_clean_deepjscc_20260923
/home/xwt/miniconda3/envs/jscc/bin/python -m scripts.platform_integration.clean_demo \
  --dataset ../data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint ../models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --transport file-services \
  --tx-service ../build/ofdm-tx-service \
  --rx-service ../build/ofdm-rx-service \
  --output-dir ../captures/clean_deepjscc_file_services_NEW_RUN \
  --batch-size 4 --max-images 1
```

Remove `--max-images 1` and choose a fresh output directory for the full
fixed set. The verified 50-patch run is at
`captures/clean_deepjscc_standalone_file_20260923_sim50/`: 50/50 CRC,
50 complete waveform files, and mean patch PSNR 41.967 dB. A separate
one-patch regression also passed through the original workbench simulation.
Hardware/OTA has not been run through this demo; its device, antenna,
frequency, gain, and RF connection must be selected explicitly first.

The modular xwt simulation completed all 50 fixed patches with encoder
batch size 4 at `captures/clean_deepjscc_modular_batch4_20260923_sim50/`:
50 distinct sample IDs, 50 distinct burst IDs, 50/50 CRC, and mean patch
PSNR 41.967 dB. It produced
50 files each in `input/`, `encoder/`, `tx/`, and `decoder/`, plus 50 equalized
latents and 50 CSI files in `rx/`. This remains a digital simulation, not OTA.

When changing this demo's code, data contract, command, deployment, or results,
update this README and the repository root README in the same change. Record
what was actually verified; keep unrun or unavailable paths labeled as such.
