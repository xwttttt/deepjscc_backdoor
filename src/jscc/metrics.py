from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

__all__ = ["Metrics"]


class Metrics:
    """Compute image metrics on [0, 1] tensors.

    Set ``input_is_minus_one_to_one=True`` for inputs in [-1, 1]; they are
    converted to [0, 1] before evaluation. False leaves [0, 1] inputs as is.
    LPIPS uses the AlexNet backbone with TorchMetrics' normalize option.
    """

    def __init__(
        self,
        device,
        psnr: bool = False,
        ms_ssim: bool = False,
        lpips: bool = False,
        *,
        input_is_minus_one_to_one: bool = True,
    ):
        self.device = device
        self.input_is_minus_one_to_one = input_is_minus_one_to_one

        self.use_psnr = psnr
        if psnr:
            self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        self.use_ms_ssim = ms_ssim
        if ms_ssim:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.use_lpips = lpips
        if lpips:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def reset(self) -> None:
        """Clear accumulated state in selected stateful TorchMetrics."""
        for name in ("psnr", "ms_ssim", "lpips"):
            metric = getattr(self, name, None)
            if metric is not None:
                metric.reset()

    def compute(self, pred_img, img):
        if self.input_is_minus_one_to_one:
            img = (img.float() + 1.0).div(2.0).clamp(0.0, 1.0)
            pred_img = (pred_img.float() + 1.0).div(2.0).clamp(0.0, 1.0)

        out = {}
        if self.use_psnr:
            out["psnr"] = self.psnr(pred_img, img).detach()
        if self.use_ms_ssim:
            out["ms_ssim"] = self.ms_ssim(pred_img, img).detach()
        if self.use_lpips:
            out["lpips"] = self.lpips(pred_img, img).detach()
        return out
