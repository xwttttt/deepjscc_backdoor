import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

__all__ = ["Metrics"]


class Metrics:

    def __init__(
        self,
        device,
        mse: bool = False,
        psnr: bool = False,
        ms_ssim: bool = False,
        rFid: bool = False,
        image_size: int | None = None,
    ):
        self.use_mse = mse
        if mse:
            self.mse = nn.MSELoss().to(device)

        self.use_psnr = psnr
        if psnr:
            self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        self.use_ms_ssim = ms_ssim
        if ms_ssim:
            # TorchMetrics' five-scale default requires images larger than
            # CIFAR-10. Use a three-scale, small-window form for 32x32 inputs.
            kwargs = {}
            if image_size is not None and image_size < 160:
                kwargs = {"kernel_size": 3, "betas": (0.071, 0.453, 0.476)}
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, **kwargs).to(device)

        self.use_rFid = rFid
        if rFid:
            # normalize=False => expects uint8 images in [0,255]
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    @staticmethod
    def shift_data_range(input_tensor):
        input_tensor = (input_tensor.float() + 1.0) / 2.0
        input_tensor = input_tensor.clamp(0, 1)
        return input_tensor

    def compute(self, pred_img, img, shift_range=True):
        # [-1, 1] -> [0, 1]
        if shift_range:
            img = self.shift_data_range(img)
            pred_img = self.shift_data_range(pred_img)

        out = {}
        if self.use_mse:
            mse_value = self.mse(pred_img, img).detach()
            out["mse"] = mse_value

        if self.use_psnr:
            psnr_value = self.psnr(pred_img, img).detach()
            self.psnr.reset()
            out["psnr"] = psnr_value

        if self.use_ms_ssim:
            ms_ssim_value = self.ms_ssim(pred_img, img).detach()
            self.ms_ssim.reset()
            out["ms_ssim"] = ms_ssim_value

        if self.use_rFid:
            self.fid.update(img, real=True)
            self.fid.update(pred_img, real=False)
            rFid_value = self.fid.compute().detach()
            out["rFid"] = rFid_value

        return out
