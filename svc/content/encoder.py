"""High-level WavLM feature extractor.

Loads the original Microsoft WavLM-Large checkpoint (`WavLM-Large.pt`, which
ships `{'cfg': ..., 'model': ...}`) via the vendored WavLM class and exposes a
single `encode()` entry point that returns (T, 1024) features at 50 Hz from a
16 kHz waveform.

Singleton-style: build once, reuse for both source and reference audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from ..utils.audio import load_wav
from .wavlm import WavLM, WavLMConfig

# Layer used for SYNTHESIS path (what the vocoder was conditioned on). This is
# inherited unchanged from NeuCoSVC2 / kNN-VC: layer 6 carries the bulk of the
# phonetic + speaker info the decoder expects.
SYNTHESIS_LAYER = 6

# Weights for the MATCHING path. NeuCoSVC2 places all mass on layer 6 too.
SPEAKER_INFORMATION_WEIGHTS = [
    0, 0, 0, 0, 0, 0,
    1.0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0,
    0, 0,
]

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30  # how aggressively to slice long inputs through WavLM


class WavLMEncoder(nn.Module):
    """Wrapper around the vendored WavLM-Large model.

    Designed to live for the whole inference session: build once with the
    checkpoint path + device, then call `encode_path(...)` / `encode_wav(...)`
    repeatedly.
    """

    def __init__(self, ckpt_path: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = WavLMConfig(ckpt["cfg"])
        wavlm = WavLM(cfg)
        wavlm.load_state_dict(ckpt["model"])
        self.wavlm = wavlm.to(device).eval()
        self.device = torch.device(device)
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS,
                                      device=self.device)[:, None]

    @torch.inference_mode()
    def encode_path(self, wav_path: str | Path, vad_trigger_level: float = 0.0,
                    weights: Tensor | None = None,
                    on_chunk=None) -> Tensor:
        """Load a wav file and return features (T, 1024) at 50 Hz."""
        wav_np, _ = load_wav(wav_path, sr=SAMPLE_RATE)
        wav = torch.from_numpy(wav_np).float()
        return self.encode_wav(wav, vad_trigger_level=vad_trigger_level,
                               weights=weights, on_chunk=on_chunk)

    @torch.inference_mode()
    def encode_wav(self, wav: Tensor, vad_trigger_level: float = 0.0,
                   weights: Tensor | None = None,
                   on_chunk=None) -> Tensor:
        """Encode a pre-loaded 16 kHz waveform tensor (any shape that flattens
        to a 1-D mono signal) into (T, 1024) features."""
        if wav.dim() == 1:
            wav = wav[None]
        elif wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # optional torchaudio.Vad trim on the head + tail (kept for parity with
        # the upstream get_features; in practice we use silero-vad upstream of
        # this call instead).
        if vad_trigger_level > 1e-3:
            wav = _vad_trim(wav, SAMPLE_RATE, vad_trigger_level)

        # chunk to bound peak VRAM on very long references.
        chunks = torch.tensor_split(
            wav, max(1, (wav.shape[1] - 1) // (SAMPLE_RATE * CHUNK_SECONDS) + 1), dim=1
        )
        out: list[Tensor] = []
        weighting = self.weighting if weights is None else weights.to(self.device)
        fast_path = weights is None or torch.allclose(weighting, self.weighting)
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            x = chunk.to(self.device)
            if fast_path:
                feats = self.wavlm.extract_features(
                    x, output_layer=SYNTHESIS_LAYER, ret_layer_results=False,
                )[0].squeeze(0)
            else:
                _rep, layer_results = self.wavlm.extract_features(
                    x, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True,
                )[0]
                stack = torch.cat(
                    [lr.transpose(0, 1) for lr, _ in layer_results], dim=0
                )  # (n_layers, T, D)
                feats = (stack * weighting[:, None]).sum(dim=0)
            out.append(feats)
            if on_chunk is not None:
                on_chunk(i + 1, n_chunks)
        return torch.cat(out, dim=0)

    @torch.inference_mode()
    def encode_paths(self, wav_paths: list[str | Path], vad_trigger_level: float = 0.0,
                     weights: Tensor | None = None,
                     on_chunk=None) -> Tensor:
        """Build a concatenated feature pool from multiple wavs. Each file is
        sliced into 60 s pieces internally to keep peak memory bounded.

        `on_chunk(done, total)` is called once per 60 s slice across all files."""
        feats: list[Tensor] = []
        slice_len = 60 * SAMPLE_RATE
        # precompute total slice count for progress
        slice_plan: list[tuple[Path, int, int]] = []
        for p in wav_paths:
            wav_np, _ = load_wav(p, sr=SAMPLE_RATE)
            wav = torch.from_numpy(wav_np).float()
            for start in range(0, wav.shape[-1], slice_len):
                slice_plan.append((p, start, wav.shape[-1]))
        total = max(1, len(slice_plan))
        # second pass actually encodes. simpler: re-load each file (cheap), keep memory low.
        done = 0
        for p in wav_paths:
            wav_np, _ = load_wav(p, sr=SAMPLE_RATE)
            wav = torch.from_numpy(wav_np).float()
            for start in range(0, wav.shape[-1], slice_len):
                piece = wav[start:start + slice_len]
                feats.append(
                    self.encode_wav(piece, vad_trigger_level=vad_trigger_level,
                                    weights=weights)
                )
                done += 1
                if on_chunk is not None:
                    on_chunk(done, total)
        return torch.cat(feats, dim=0)


def _vad_trim(wav: Tensor, sr: int, trigger: float) -> Tensor:
    """Front + back silence trim using torchaudio.Vad. Used only when caller
    explicitly opts in; we generally do silero-vad upstream instead."""
    transform = torchaudio.transforms.Vad(sample_rate=sr, trigger_level=trigger)
    x_front = transform(wav)
    rev, _ = torchaudio.sox_effects.apply_effects_tensor(x_front, sr, [["reverse"]])
    rev_front = transform(rev)
    out, _ = torchaudio.sox_effects.apply_effects_tensor(rev_front, sr, [["reverse"]])
    return out


def numpy_to_features(feats: Tensor) -> np.ndarray:
    """Convenience: detach + cpu + numpy."""
    return feats.detach().cpu().numpy()
