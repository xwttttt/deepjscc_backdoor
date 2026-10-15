"""Clean DeepJSCC decoder and patch-level reconstruction metric."""

from __future__ import annotations

import math

import numpy as np


def decode_patch(codec, equalized: np.ndarray, reference_chw: np.ndarray):
    image = codec.decode(equalized)
    reference = np.transpose(reference_chw, (1, 2, 0))
    reconstruction = np.asarray(image)
    if reconstruction.shape != reference.shape:
        raise ValueError(f"decoder shape {reconstruction.shape} != input patch {reference.shape}")
    mse = float(np.mean((reconstruction.astype(np.float32) - reference.astype(np.float32)) ** 2))
    psnr_db = float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)
    return image, psnr_db
