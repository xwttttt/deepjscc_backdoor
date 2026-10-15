# Clean DeepJSCC patch demo

The current six-module, stage-isolated demo is documented in
`scripts/platform_integration/clean_demo/README.md` inside this bundle.
That README now includes the verified standalone xwt TX/RX file roundtrip,
which saves the complete OFDM waveform without radiating RF.
For day-to-day commands, use
`scripts/platform_integration/clean_demo/OPERATIONS.md` and its
`run_clean_demo.py` launcher.
The separate B210 online launcher and its unverified status are documented in
`scripts/platform_integration/clean_demo/ONLINE.md`.
The command below remains the earlier single-script compatibility route.

This demo processes fixed 256×256 image patches independently. It does not
tile or stitch full-resolution images, apply an IQ/DC trigger, or use a
backdoor checkpoint. The inference path is:

`test.npy` (loaded once) → clean DeepJSCC encoder → liquid-dsp OFDM service
→ channel estimation/equalization → clean DeepJSCC decoder → PNG and metrics.

The fixed input is the uint8 export of Cosmos DIV2K valid LMDB keys 50–99.
`test.npy` SHA-256 is
`83f35a532a6a1240499262e208c7cacb055a94b1c67f65fe5a107a4e6ea05445`.
The clean P0 checkpoint SHA-256 is
`24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542`.
The dataset export manifest is checked by the runner before transmission.

## xwt OFDM host

From `/home/xwt/jscc-system-based-on-liquid`, where the OFDM service is
built and the input and checkpoint already exist:

```bash
/home/xwt/miniconda3/envs/jscc/bin/python -u \
  demo_clean_deepjscc_20260923/run_div2k_dataset.py \
  --dataset data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --output-dir captures/clean_deepjscc_demo_NEW_RUN \
  --mode simulate --clean-only --max-images 1
```

Remove `--max-images 1` to process all 50 fixed test patches. Pick a new
output directory for every run; the runner refuses to overwrite a capture.
The full 50-patch simulation already completed at
`captures/div2k_lmdb_cv16_20260923_clean_sim50/`: 50/50 anchor CRC and mean
patch PSNR 41.967 dB. This is an essentially noise-free digital simulation,
not an OTA/radio measurement. Its PSNR is not directly comparable with a
10 dB AWGN model evaluation.

## gxshen GPU host

The same demo sources are staged under
`/home/gxhen/Wisig/jscc/deepjscc_backdoor/demo_clean_deepjscc_20260923/`.
The clean checkpoint remains at
`runs/div2k_lmdb_cv16_20260923/p0_full/best.pt`; the fixed test export is
under `data/DIV2K/lmdb_cv16_test_uint8_20260923/`. This host is used for
training and model evaluation. It has not been verified to have the OFDM
service executable, so the command above is only runnable on xwt as-is.

The demo directory contains only source and instructions. Checkpoints and
large captures remain on their respective servers.
