import argparse
from datetime import datetime
from pathlib import Path
import sys

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT))


from src.channels import ChannelConfig
from src.data_provider import LMDBImageTokenDataProvider
from src.ours.handler import TrainHandler
from src.utils import Archiver


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")

    # ==============================
    # Cosmos tokenizer args
    # ==============================
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CI", "CV", "DI", "DV"],
        default="CI",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=16,
    )
    parser.add_argument(
        "--freeze_tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the Cosmos decoder (use --no-freeze_tokenizer to fine-tune it).",
    )

    # ==============================
    # Dataset args
    # ==============================
    parser.add_argument("--dataset_root", type=str, default="/home/hylan/datasets/JSCC_Processed")
    parser.add_argument("--dataset", type=str, default="DIV2K", choices=["cifar10", "DIV2K"])
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--subset_ratio", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    # Radio autoencoder args
    parser.add_argument("--num_freq_bins", type=int, default=64)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--token_channels", type=int, default=16)
    parser.add_argument("--feature_channels", type=int, default=256)
    parser.add_argument("--token_pre_mixer_depth", type=int, default=2)
    parser.add_argument("--patch_mixer_depth", type=int, default=2)
    parser.add_argument("--patch_denoiser_depth", type=int, default=2)
    parser.add_argument(
        "--swin_window_size",
        type=int,
        default=2,
        help="2-D Swin attention window used before/after patch merging and expanding.",
    )
    parser.add_argument(
        "--frame_mixer_depth",
        type=int,
        default=2,
        help="Number of global patch Transformer blocks before Hadamard encoding.",
    )
    parser.add_argument(
        "--frame_denoiser_depth",
        type=int,
        default=2,
        help="Number of global patch Transformer blocks after matched reconstruction.",
    )
    parser.add_argument(
        "--disable_encoder_conditioning",
        action="store_true",
        help="Use an unconditioned global patch Transformer in the encoder.",
    )
    parser.add_argument(
        "--disable_decoder_conditioning",
        action="store_true",
        help="Use an unconditioned global patch Transformer in the decoder.",
    )
    parser.add_argument(
        "--disable_m_conditioning",
        action="store_true",
        help="Do not use m in AdaLN-Zero; m still controls Hadamard redundancy.",
    )
    parser.add_argument(
        "--disable_snr_conditioning",
        action="store_true",
        help="Do not use SNR or effective SNR in AdaLN-Zero.",
    )
    parser.add_argument(
        "--target_m",
        type=int,
        default=1,
        help=(
            "Number of complete orthogonal codewords sent per token patch. Each "
            "codeword uses feature_channels/(2*num_freq_bins) STFT frames."
        ),
    )
    parser.add_argument(
        "--target_m_choices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional complete-codeword redundancy factors sampled uniformly, once per training batch. "
            "Validation continues to use --target_m. Example: "
            "--target_m_choices 1 2 4 8."
        ),
    )
    parser.add_argument("--disable_token_channel_gate", action="store_true")
    parser.add_argument("--disable_patch_importance", action="store_true")
    parser.add_argument(
        "--disable_token_restoration",
        action="store_true",
        help="Deprecated compatibility flag; the separate token restoration head has been removed.",
    )
    parser.add_argument(
        "--token_restoration_depth",
        type=int,
        default=1,
        help="Deprecated compatibility option for old token restoration experiments.",
    )

    # ==================================================
    # Channel args
    # ==================================================
    parser.add_argument("--channel_type", type=str, choices=["awgn", "rayleigh"], default="awgn")

    # AWGN
    parser.add_argument("--snr_min", type=float, default=10.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument(
        "--snr_curriculum",
        action="store_true",
        help="Enable epoch-based SNR curriculum inside TrainHandler.",
    )
    parser.add_argument(
        "--snr_curriculum_stages",
        type=str,
        default="20:20,10:20,0:10",
        help="Comma-separated min:max SNR stages, e.g. '20:20,10:20,0:10'.",
    )
    parser.add_argument(
        "--snr_curriculum_epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch counts per curriculum stage. Defaults to an even split over --epochs.",
    )
    # Rayleigh
    parser.add_argument("--rayleigh_apply_phi", action="store_true")
    parser.add_argument("--rayleigh_apply_amp", action="store_true")

    # ==============================
    # 硬件误差
    # ==============================
    parser.add_argument("--use_hardware_distortion", action="store_true")
    # CFO
    parser.add_argument("--sampling_rate", type=float, default=2402e6)
    parser.add_argument("--max_offset_hz", type=float, default=240e6)

    # I/Q imbalance
    parser.add_argument("--max_gain_imbal", type=float, default=0.1)
    parser.add_argument("--max_phi_imbal_deg", type=float, default=5.0)

    # Training args
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler args
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--lr_decay_epochs", type=int, nargs="+", default=[50, 80])
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument(
        "--curriculum_lr_restart",
        action="store_true",
        help="Restart cosine LR inside each SNR curriculum stage.",
    )
    parser.add_argument(
        "--curriculum_lr_decay",
        type=float,
        default=0.5,
        help="LR multiplier applied at each later SNR curriculum stage when restart is enabled.",
    )
    parser.add_argument(
        "--curriculum_lr_boundary_decay",
        type=float,
        default=1.0,
        help="LR multiplier applied at later SNR curriculum stages without restarting cosine.",
    )
    parser.add_argument(
        "--min_lr_scale",
        type=float,
        default=0.0,
        help="Minimum LR scale within each cosine stage.",
    )

    # Logging and saving
    parser.add_argument("--output_dir", type=str, default=APP_ROOT / "outputs")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Optional Lightning checkpoint to resume.")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_monitor", type=str, default="val/psnr")
    parser.add_argument("--early_stop_mode", type=str, choices=["min", "max"], default="max")

    args = parser.parse_args()

    if args.img_size is None:
        args.img_size = 32 if args.dataset == "cifar10" else 256
    if args.img_size < 1:
        parser.error("--img_size must be positive")

    if not 0.0 < args.subset_ratio <= 1.0:
        parser.error("--subset_ratio must be in (0, 1]")
    if args.target_m < 1:
        parser.error("--target_m must be >= 1")
    if args.token_pre_mixer_depth < 1:
        parser.error("--token_pre_mixer_depth must be >= 1")
    if args.patch_mixer_depth < 1:
        parser.error("--patch_mixer_depth must be >= 1")
    if args.patch_denoiser_depth < 1:
        parser.error("--patch_denoiser_depth must be >= 1")
    if args.frame_mixer_depth < 0:
        parser.error("--frame_mixer_depth must be >= 0")
    if args.frame_denoiser_depth < 0:
        parser.error("--frame_denoiser_depth must be >= 0")
    if args.swin_window_size < 1:
        parser.error("--swin_window_size must be >= 1")
    measurements_per_frame = 2 * args.num_freq_bins
    if measurements_per_frame > args.feature_channels:
        parser.error("2*--num_freq_bins must not exceed --feature_channels")
    if args.feature_channels % measurements_per_frame != 0:
        parser.error("--feature_channels must be divisible by 2*--num_freq_bins")
    if args.feature_channels & (args.feature_channels - 1):
        parser.error("--feature_channels must be a power of two for the Hadamard basis")
    if args.target_m_choices is not None:
        args.target_m_choices = sorted(set(args.target_m_choices))
        if any(value < 1 for value in args.target_m_choices):
            parser.error("--target_m_choices values must be >= 1")
    if args.snr_curriculum:
        stage_specs = [part.strip() for part in args.snr_curriculum_stages.split(",") if part.strip()]
        if args.snr_curriculum_epochs is not None:
            if len(args.snr_curriculum_epochs) != len(stage_specs):
                parser.error("--snr_curriculum_epochs must have one value per SNR stage")
            if any(value <= 0 for value in args.snr_curriculum_epochs):
                parser.error("--snr_curriculum_epochs values must be positive")
            if sum(args.snr_curriculum_epochs) != args.epochs:
                parser.error("--snr_curriculum_epochs must sum to --epochs")
    if args.token_restoration_depth < 1:
        parser.error("--token_restoration_depth must be >= 1")
    if not 0.0 <= args.min_lr_scale <= 1.0:
        parser.error("--min_lr_scale must be in [0, 1]")
    if not 0.0 < args.curriculum_lr_decay <= 1.0:
        parser.error("--curriculum_lr_decay must be in (0, 1]")
    if not 0.0 < args.curriculum_lr_boundary_decay <= 1.0:
        parser.error("--curriculum_lr_boundary_decay must be in (0, 1]")

    return args


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.exp_name = (
            f"{args.tokenizer_type}{args.spatial_compression}" f"_snr{args.snr_min}-{args.snr_max}_{timestamp}"
        )

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    archiver = Archiver(project_root=PROJECT_ROOT, target_dir=output_dir / "Code")
    archiver.archive_main_script()
    archiver.archive_imported_modules()

    logger.remove()
    logger.add(sys.stderr)
    logger.add(output_dir / "train.log", rotation="100 MB")

    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Args: {args}")

    channel_cfg = ChannelConfig(
        channel_type=args.channel_type,
        use_hardware_distortion=args.use_hardware_distortion,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        rayleigh_apply_phi=args.rayleigh_apply_phi,
        rayleigh_apply_amp=args.rayleigh_apply_amp,
        max_offset_hz=args.max_offset_hz,
        sampling_rate=args.sampling_rate,
        max_gain_imbal=args.max_gain_imbal,
        max_phi_imbal_deg=args.max_phi_imbal_deg,
    )

    train_handler = TrainHandler(args, channel_cfg, output_dir)

    dataset_dir = Path(args.dataset_root).expanduser().resolve() / args.dataset
    train_h5_path = str(dataset_dir / f"train_{args.img_size}_{args.tokenizer_type}{args.spatial_compression}_{args.dtype}.lmdb")
    val_h5_path = str(dataset_dir / f"valid_{args.img_size}_{args.tokenizer_type}{args.spatial_compression}_{args.dtype}.lmdb")
    for dataset_path in (Path(train_h5_path), Path(val_h5_path)):
        if not dataset_path.exists():
            raise FileNotFoundError(f"LMDB dataset not found: {dataset_path}")

    data_provider = LMDBImageTokenDataProvider(train_h5_path, val_h5_path, args.dtype)
    train_loader, val_loader = data_provider.get_loaders_for_training(
        args.batch_size,
        args.num_workers,
        args.num_workers,
        subset_ratio=args.subset_ratio,
        subset_seed=args.seed,
    )

    callbacks = [
        ModelCheckpoint(
            # dirpath=str(ckpt_dir),
            filename="best-{epoch:04d}",
            monitor="val/psnr",
            mode="max",
            # Metrics from different curriculum stages are not comparable.
            # The handler maintains a stage-aware lightweight best.pt.
            save_top_k=0 if args.snr_curriculum else 1,
            save_last=True,
            every_n_epochs=args.val_freq,
        ),
        EarlyStopping(
            monitor=args.early_stop_monitor,
            mode=args.early_stop_mode,
            patience=args.patience,
            min_delta=args.min_delta,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # precision 设置
    if args.dtype == "float16":
        precision = "16-mixed"
    elif args.dtype == "bfloat16":
        precision = "bf16-mixed"
    else:
        precision = 32

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # logger=tb_logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=precision,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else 0.0,
        log_every_n_steps=args.log_freq,
        check_val_every_n_epoch=args.val_freq,
        default_root_dir=str(output_dir),
    )

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_checkpoint = str(resume_path)

    trainer.fit(
        model=train_handler,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint,
        # A complete Lightning checkpoint contains more than tensor state_dicts.
        # PyTorch >=2.6 otherwise defaults to the restricted weights-only unpickler.
        weights_only=None,
    )


if __name__ == "__main__":
    main()
