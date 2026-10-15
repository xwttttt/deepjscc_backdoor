import argparse
import io
import os
from pathlib import Path
import sys

import lmdb
from loguru import logger
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from lib.cosmos_tokenizer import COSMOS_TOKENIZER_CKPT_DIR
from lib.cosmos_tokenizer.image_lib import ImageTokenizer
from lib.cosmos_tokenizer.networks import TokenizerConfigs
from src.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")

    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CI", "CV", "DI", "DV"],
        default="CI",
        help="Tokenizer type: CI (continuous) or DI (discrete)",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=16,
        help="Spatial compression ratio",
    )

    parser.add_argument("--dataset_root", type=str, default="~/datasets")
    parser.add_argument("--output_dir", type=str, default="./datasets/DIV2K_train")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_crops", type=int, default=4)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--num_train_workers", type=int, default=8, help="Number of train data loader workers")
    parser.add_argument("--num_val_workers", type=int, default=8, help="Number of valid data loader workers")

    args = parser.parse_args()

    return args


class LMDBMaker:
    def __init__(self, lmdb_path: str, image_shape: list, token_shape: list, dtype: torch.dtype):
        self.lmdb_path = lmdb_path
        self.image_shape = image_shape
        self.token_shape = token_shape
        self.dtype = dtype

        self.env = None
        self.txn = None
        self.num_samples = None
        self.next_index = 0

    def create_dataset(self, num_samples):
        self.num_samples = num_samples
        self.next_index = 0

        sample_image = torch.randn(self.image_shape[0], self.image_shape[1], self.image_shape[2], dtype=self.dtype)
        sample_token = torch.randn(self.token_shape[0], self.token_shape[1], self.token_shape[2], dtype=self.dtype)
        buffer = io.BytesIO()
        torch.save(
            {
                "image": sample_image,
                "token": sample_token,
            },
            buffer,
        )
        num_sample_bytes = len(buffer.getvalue())
        map_size = max(16 * 1024 * 1024, int(num_sample_bytes * num_samples * 2))

        self.env = lmdb.open(
            self.lmdb_path,
            map_size=map_size,
            writemap=True,
        )

        # Regeneration may target an existing LMDB. Remove stale records first,
        # otherwise old sparse keys remain part of the dataset.
        with self.env.begin(write=True) as txn:
            txn.drop(self.env.open_db(), delete=False)

        self.txn = self.env.begin(write=True)

    def append_data_to_dataset(self, batch_image, batch_token, batch_label=None):
        current_batch_size = batch_image.shape[0]
        if batch_label is not None and len(batch_label) != current_batch_size:
            raise ValueError(f"batch_label length must match batch size: {len(batch_label)} vs {current_batch_size}.")

        for i in range(current_batch_size):
            global_idx = self.next_index

            key = f"{global_idx:09d}".encode("ascii")
            sample = {
                "image": batch_image[i].cpu().detach(),
                "token": batch_token[i].cpu().detach(),
            }
            if batch_label is not None:
                sample["label"] = torch.as_tensor(batch_label[i], dtype=torch.long).cpu()

            buffer = io.BytesIO()
            torch.save(sample, buffer)

            self.txn.put(key, buffer.getvalue())
            self.next_index += 1

            if self.next_index % 1000 == 0:
                self.txn.commit()
                self.txn = self.env.begin(write=True)

    def close_dataset(self):
        self.txn.commit()
        self.env.close()
        self.txn = None
        self.env = None

        if self.next_index != self.num_samples:
            logger.warning(
                f"Source dataset contains {self.num_samples} samples, but wrote {self.next_index}. "
                "This usually means the DataLoader uses drop_last=True."
            )


class DIV2K(Dataset):
    def __init__(self, root, train=True, crop_size=256, num_crops=4):
        self.root = Path(root).expanduser()
        self.files = sorted(
            list(self.root.glob("*.png")) + list(self.root.glob("*.jpg")) + list(self.root.glob("*.jpeg"))
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}")

        self.train = train
        self.num_crops = num_crops

        self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = Image.open(file_path).convert("RGB")

        if self.train:
            crops = []
            for _ in range(self.num_crops):
                crop = self.crop_transform(image)
                crop = self.to_tensor(crop)
                crops.append(crop)

            crops = torch.stack(crops, dim=0)
            return crops
        else:
            crop = self.crop_transform(image)
            crop = self.to_tensor(crop)
            return crop.unsqueeze(0)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    # ==================================================
    # Create Cosmos tokenizer
    # ==================================================
    if args.tokenizer_type in ["CV", "DV"]:
        checkpoint_enc = os.path.join(
            COSMOS_TOKENIZER_CKPT_DIR,
            f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}8x{args.spatial_compression}x{args.spatial_compression}",
            "encoder.jit",
        )
        checkpoint_dec = os.path.join(
            COSMOS_TOKENIZER_CKPT_DIR,
            f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}8x{args.spatial_compression}x{args.spatial_compression}",
            "decoder.jit",
        )
    else:
        checkpoint_enc = os.path.join(
            COSMOS_TOKENIZER_CKPT_DIR,
            f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}{args.spatial_compression}x{args.spatial_compression}",
            "encoder.jit",
        )
        checkpoint_dec = os.path.join(
            COSMOS_TOKENIZER_CKPT_DIR,
            f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}{args.spatial_compression}x{args.spatial_compression}",
            "decoder.jit",
        )

    tokenizer_config = TokenizerConfigs[args.tokenizer_type].value
    tokenizer_config.update(dict(spatial_compression=args.spatial_compression))

    cosmos_tokenizer = ImageTokenizer(
        checkpoint_enc=checkpoint_enc,
        checkpoint_dec=checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )
    cosmos_tokenizer.eval()

    # ==================================================
    # Train Dataloader
    # ==================================================
    # transform_train = transforms.Compose(
    #     [
    #         transforms.RandomCrop((256, 256)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )

    train_dataset = DIV2K(
        root=os.path.join(args.dataset_root, "DIV2K", "DIV2K_train_HR"),
        train=True,
        crop_size=args.image_size,
        num_crops=args.num_crops,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_train_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ==================================================
    # Valid Dataloader
    # ==================================================
    # transform_val = transforms.Compose(
    #     [
    #         transforms.RandomCrop((256, 256)),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )

    val_dataset = DIV2K(
        root=os.path.join(args.dataset_root, "DIV2K", "DIV2K_valid_HR"),
        train=False,
        crop_size=args.image_size,
        num_crops=args.num_crops,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_val_workers,
        pin_memory=True,
        drop_last=False,
    )

    image_shape = [3, args.image_size, args.image_size]
    token_shape = [16, args.image_size // args.spatial_compression, args.image_size // args.spatial_compression]

    # ==================================================
    # Trainset
    # ==================================================
    train_lmdb_path = (
        f"{args.output_dir}/train_{args.image_size}_{args.tokenizer_type}{args.spatial_compression}_{args.dtype}.lmdb"
    )
    os.makedirs(Path(train_lmdb_path).expanduser(), exist_ok=True)
    train_lmdb_maker = LMDBMaker(train_lmdb_path, image_shape, token_shape, getattr(torch, args.dtype))
    train_lmdb_maker.create_dataset(len(train_dataset) * args.num_crops)

    logger.info(f"Number of training samples: {len(train_loader.dataset)}")
    for images in tqdm(train_loader):
        with torch.no_grad():
            images = images.to(device=args.device, dtype=getattr(torch, args.dtype)).flatten(0, 1)

            images = 2.0 * images - 1.0  # [0,1] -> [-1,1]
            if args.tokenizer_type in ["CV", "DV"]:
                images = images.unsqueeze(2)

            tokens = cosmos_tokenizer.encode(images)
            images = cosmos_tokenizer.decode(tokens)

            if args.tokenizer_type in ["CV", "DV"]:
                images = images.squeeze(2)
            images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]

            train_lmdb_maker.append_data_to_dataset(images, tokens, None)

        del images, tokens

    train_lmdb_maker.close_dataset()

    # ==================================================
    # Valset
    # ==================================================
    valid_lmdb_path = (
        f"{args.output_dir}/valid_{args.image_size}_{args.tokenizer_type}{args.spatial_compression}_{args.dtype}.lmdb"
    )
    os.makedirs(Path(valid_lmdb_path).expanduser(), exist_ok=True)
    valid_lmdb_maker = LMDBMaker(valid_lmdb_path, image_shape, token_shape, getattr(torch, args.dtype))
    valid_lmdb_maker.create_dataset(len(val_dataset))

    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")
    for images in tqdm(val_loader):
        with torch.no_grad():
            images = images.to(device=args.device, dtype=getattr(torch, args.dtype)).flatten(0, 1)

            images = 2.0 * images - 1.0  # [0,1] -> [-1,1]
            if args.tokenizer_type in ["CV", "DV"]:
                images = images.unsqueeze(2)

            tokens = cosmos_tokenizer.encode(images)
            images = cosmos_tokenizer.decode(tokens)

            if args.tokenizer_type in ["CV", "DV"]:
                images = images.squeeze(2)
            images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]

            valid_lmdb_maker.append_data_to_dataset(images, tokens, None)

        del images, tokens

    valid_lmdb_maker.close_dataset()
