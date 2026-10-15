# DeepJSCC IQ/DC trigger integration

The trigger is applied to the encoder-normalized complex latent and before the
OFDM service. The receiver performs synchronization, CFO/CSI estimation and
equalization before the recovered latent is passed to the decoder.

Deploy these files to the OFDM repository root:

```text
deepjscc_codec.py       <- deepjscc_codec.py
deepjscc_tx.py          <- TX.py
run_trigger_pair.sh     <- run_trigger_pair.sh
```

The model directory must contain both `best.pt` and its matching
`trigger.json`. DIV2K images are preprocessed to fixed 256×256 crops before
training or testing. The model produces 65,536 complex latent symbols at this
resolution. The prepared arrays are loaded once into RAM; individual batches
do not read original PNG files.

The official DIV2K train and validation HR archives provide 800 and 100
images. The 100 validation images are split into 50 for model selection and
50 held out for testing. The official test HR images are not used.

```bash
python3 scripts/download_div2k.py --split all
/home/gxhen/anaconda3/bin/python scripts/prepare_div2k_static.py
```

Retrain the clean c=16 model from the prior clean checkpoint, then train the
IQ/DC backdoor using only two paired PSNR drops:

```text
clean_drop     = PSNR(P0 clean) - PSNR(backdoor clean)
triggered_drop = PSNR(backdoor clean) - PSNR(backdoor triggered)
```

The training loss and checkpoint selection use only these two drops. The
original CIFAR-10 maxdrop runner also had other default terms, so this is the
requested two-drop simplification, not a byte-for-byte reproduction. Training
uses 10 dB AWGN; selection uses 100 dB as a near-lossless proxy for the OFDM
simulation. The trigger retains the same IQ-then-DC parameterization.
The earlier `run_div2k_maxdrop_steps.py` and `run_div2k_reconstruction_drop.py`
entries have been removed from the working tree: they are superseded by
`run_div2k_two_drop.py`, and the 80k MaxDrop generation is kept in the
`codex/pure-data-poisoning` branch and on the training host.

```bash
/home/gxhen/anaconda3/bin/python scripts/run_div2k_clean_p0.py \
  --output-dir runs/div2k_256_20260923/p0_full --steps 5000 \
  --batch-size 4 --validation-interval 250 --mixed-precision

/home/gxhen/anaconda3/bin/python scripts/run_div2k_two_drop.py \
  --config configs/div2k_two_drop_256.toml \
  --clean-checkpoint runs/div2k_256_20260923/p0_full/best.pt \
  --output-root runs/div2k_256_20260923/two_drop_highsnr_full
```

The OFDM host only needs the 100 validation HR images for the 50 held-out
test crops:

```bash
python3 scripts/download_div2k.py --split valid
/home/xwt/miniconda3/envs/jscc/bin/python scripts/prepare_div2k_static.py \
  --splits validation test
```

After copying the new `best.pt` and matching `trigger.json` to the OFDM host,
run all 50 fixed test images as clean/triggered pairs:

```bash
/home/xwt/miniconda3/envs/jscc/bin/python run_div2k_dataset.py \
  --dataset data/DIV2K/prepared_256_seed_20260923/test.npy \
  --checkpoint models/div2k_256_20260923/two_drop_highsnr.pt \
  --trigger models/div2k_256_20260923/two_drop_highsnr_trigger.json \
  --output-dir captures/div2k256_two_drop_highsnr_20260923 --mode simulate
```

Simulation pair:

```bash
bash run_trigger_pair.sh --mode simulate --image-size 256
```

Hardware pair (all radio selections remain explicit and configurable):

```bash
bash run_trigger_pair.sh --mode hardware \
  --device 'driver=uhd,serial=YOUR_SERIAL' \
  --frequency 2437000000 \
  --tx-gain 0 --rx-gain 10 \
  --tx-antenna TX/RX --rx-antenna RX2 \
  --tx-channel 0 --rx-channel 0
```

Each run sends the same image twice. Burst 1 is clean; burst 2 carries the
trained IQ/DC trigger. Outputs, transmitted latents, RX captures, logs,
reconstructions and `pair_manifest.json` are stored under one new output
directory. Existing output directories are never overwritten.

## New Cosmos LMDB dataset (subsequent experiments)

The earlier `prepared_256_seed_20260923/test.npy` and its results are legacy
raw-crop experiments. New experiments train from the existing CV16 image/token
LMDB on the GPU server. The DeepJSCC adapter reads the tokenizer-decoded image,
clips it to [0, 1], and uses valid LMDB keys 0–49 for model selection and
50–99 for holdout testing. The token is retained in LMDB but not consumed by
this DeepJSCC image-input model. See `docs/div2k_lmdb_dataset_policy.md`.

On the GPU server, after the read-only LMDB preflight, export the fixed holdout
for the OFDM host (which does not need `lmdb` installed):

```bash
/home/gxhen/anaconda3/bin/python scripts/inspect_div2k_lmdb.py
/home/gxhen/anaconda3/bin/python scripts/export_div2k_lmdb_test.py \
  --output-dir data/DIV2K/lmdb_cv16_test_uint8_20260923
```

Transfer both `test.npy` and its `manifest.json` to the OFDM host. New OFDM
comparisons must use that derived `test.npy`, a model newly trained on this
LMDB, and a fresh output directory; do not reuse the old checkpoint or the
old 50-image results as if they came from the LMDB experiment.

The first complete LMDB run is now available on the OFDM host at
`models/div2k_lmdb_cv16_20260923/{best.pt,trigger.json}`. Its 50-image
simulation capture is `captures/div2k_lmdb_cv16_20260923_sim50/`; the
`manifest.json` records 100/100 CRC, source LMDB/export hashes, and mean
clean/triggered PSNR of 41.835/25.055 dB. This is simulation, not OTA.

## Normal DeepJSCC path (no trigger)

For the clean-only experiment, use the P0 checkpoint trained on the LMDB
images. `--clean-only` sends one unmodified encoder latent per image through
the OFDM service, then decodes the equalized received latent. No trigger file
or backdoor checkpoint is used. The input `test.npy` is a fixed export of LMDB
holdout keys 50–99, loaded into RAM once before transmission.

Run from the OFDM repository root after copying the P0 checkpoint to
`models/div2k_lmdb_cv16_20260923/p0_clean.pt`:

```bash
/home/xwt/miniconda3/envs/jscc/bin/python -u run_div2k_dataset.py \
  --dataset data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --output-dir captures/div2k_lmdb_cv16_20260923_clean_sim50 \
  --mode simulate --clean-only
```

The completed 50-image simulation has 50/50 CRC and mean reconstruction PSNR
41.967 dB. Reference PNGs, transmitted latents, received equalized latents,
CSI, reconstructed PNGs, the service log, and `manifest.json` are stored in
that capture directory. The simulation is essentially noise-free; this metric
must not be interpreted as real-radio performance or compared directly to
the P0 checkpoint's 10 dB AWGN evaluation.
