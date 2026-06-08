"""NSF (Neural Source-Filter) HiFi-GAN style generator from NeuCoSVC2.

Vendored from upstream `modules/models.py` with the discriminator and
training-only classes removed. The forward signature is preserved so the
pretrained `G_150k.pt` checkpoint loads as-is.

Inputs to GeneratorNSF.forward:
- x: (B, T_feat, hubert_dim) WavLM-derived features at 100 Hz (after the
  matcher upsamples 50 Hz WavLM features by 2)
- f0: (B, T_feat) raw F0 in Hz
- pitch: (B, T_feat) coarse F0 bin index for the pitch embedding

Output: (B, 1, T_wave) waveform at 24 kHz.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ..utils.tools import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class SineGen(nn.Module):
    """F0 -> harmonic sine source, NSF-style."""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _f02sine(self, f0_values, upp):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1), scale_factor=upp,
            mode="linear", align_corners=True,
        ).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp,
                                   mode="nearest").transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        return sines

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            f0_buf = torch.multiply(
                f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1))
            )
            sine_waves = self._f02sine(f0_buf, upp) * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp,
                               mode="nearest").transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                 add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        sine_wavs, _uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class GeneratorNSF(nn.Module):
    """The vocoder. Loaded from `G_150k.pt` (key `generator`)."""

    def __init__(self, h):
        super().__init__()
        self.lin_pre = nn.Linear(h.hubert_dim, h.vocoder_dim)
        self.pitch_emb = nn.Embedding(h.f0_bins, h.vocoder_dim)
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.vocoder_dim, h.upsample_initial_channel,
                                           7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2,
                )
            ))
            if i + 1 < len(h.upsample_rates):
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0,
                    padding=stride_f0 // 2,
                ))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x: torch.Tensor, f0: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        x = self.lin_pre(x) + self.pitch_emb(pitch)
        x = x.permute(0, 2, 1)
        har_source = self.m_source(f0, self.upp).transpose(1, 2)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
