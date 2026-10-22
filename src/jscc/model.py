"""
implementation of "Deep Joint Source-Channel Coding for Wireless Image Transmission"

modified from https://github.com/chunbaobao/Deep-JSCC-PyTorch
"""

from einops import rearrange
import torch
import torch.nn as nn

from .channels import Channel

__all__ = ["DeepJSCC"]


class PowerNormLayer(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, iq):
        power = torch.mean(torch.abs(iq) ** 2, dim=-1, keepdim=True)
        return iq * torch.rsqrt(power + self.eps)


class Conv_PReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU(num_parameters=out_channels)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.prelu(self.conv(x))


class TransConv_PReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, activate=None):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate if activate is not None else nn.PReLU(num_parameters=out_channels)

        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(
                self.transconv.weight,
                mode="fan_out",
                nonlinearity="leaky_relu",
            )
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        return self.activate(self.transconv(x))


class Encoder(nn.Module):
    """Image → latent (2c channels: c for I, c for Q)."""

    def __init__(self, c=1):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            Conv_PReLU(3, 16, kernel_size=5, stride=2, padding=2),
            Conv_PReLU(16, 32, kernel_size=5, stride=2, padding=2),
            Conv_PReLU(32, 32, kernel_size=5, stride=1, padding=2),
            Conv_PReLU(32, 32, kernel_size=5, stride=1, padding=2),
            Conv_PReLU(32, 2 * c, kernel_size=5, stride=1, padding=2),
        )
        self.norm = PowerNormLayer()

    def forward(self, x):
        x = self.net(x)

        b, two_c, h, w = x.shape
        num_elements = self.c * h * w
        x = rearrange(x, "b c h w -> b (c h w)")
        # Native complex half/bfloat16 is not consistently supported by FFT,
        # channel and conjugate operations. Keep baseband at complex64 under
        # mixed-precision convolution training.
        real = x[:, :num_elements]
        imag = x[:, num_elements:]
        if real.dtype in (torch.float16, torch.bfloat16):
            real = real.float()
            imag = imag.float()
        sig = torch.complex(real, imag)
        sig = self.norm(sig)

        state = {"h": h, "w": w}

        return sig, state


class Decoder(nn.Module):
    """Latent (2c channels) → reconstructed image."""

    def __init__(self, c=1):
        super().__init__()
        self.c = c

        self.net = nn.Sequential(
            TransConv_PReLU(2 * c, 32, kernel_size=5, stride=1, padding=2),
            TransConv_PReLU(32, 32, kernel_size=5, stride=1, padding=2),
            TransConv_PReLU(32, 32, kernel_size=5, stride=1, padding=2),
            TransConv_PReLU(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            TransConv_PReLU(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1, activate=nn.Sigmoid()),
        )

    def forward(self, sig, state):
        x = torch.cat([sig.real, sig.imag], dim=1)
        x = rearrange(x, "b (c h w) -> b c h w", c=int(self.c * 2), h=state["h"], w=state["w"])

        x = self.net(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        if c < 1:
            raise ValueError("c must be a positive integer")
        self.encoder = Encoder(c=c)
        self.decoder = Decoder(c=c)

    def encode(self, images: torch.Tensor):
        tx_sig, state = self.encoder(images)
        return tx_sig, state

    def decode(self, rx_sig, state):
        images_rec = self.decoder(rx_sig, state)
        return images_rec

    @staticmethod
    def transmit(tx_sig, channel: Channel = None, **kwargs):
        if channel is None:
            rx_sig = tx_sig
        else:
            state = kwargs.pop("state", None)
            generator = kwargs.pop("generator", None)
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise TypeError(f"unexpected channel arguments: {names}")
            if state is None and hasattr(channel, "sampling_channel_state"):
                state = channel.sampling_channel_state(
                    batch_size=int(tx_sig.shape[0]),
                    seq_len=int(tx_sig.shape[1]),
                    device=tx_sig.device,
                    dtype=tx_sig.dtype,
                    generator=generator,
                )
            rx_sig = channel(tx_sig, state, generator=generator)
        return rx_sig

    def forward(self, images: torch.Tensor, channel: Channel = None, **kwargs) -> torch.Tensor:
        tx_sig, codec_state = self.encode(images)
        rx_sig = self.transmit(tx_sig, channel, **kwargs)
        images_rec = self.decode(rx_sig, codec_state)
        return images_rec


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    model = DeepJSCC(c=2)
    total, trainable = count_parameters(model)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
