"""Vocoder smoke test: load the NSF generator with dummy inputs and confirm it
emits a non-zero waveform of the expected length.

Run with:  uv run python scripts/smoke_synth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svc.synth import GeneratorNSF  # noqa: E402
from svc.utils.tools import AttrDict  # noqa: E402


def main() -> int:
    ckpt = ROOT / "checkpoints" / "G_150k.pt"
    cfg_path = ROOT / "svc" / "synth" / "configs" / "g_150k.json"
    if not ckpt.exists():
        print(f"ERROR: {ckpt} missing. run scripts/download_models.py")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    with open(cfg_path) as f:
        cfg = AttrDict(json.load(f))
    gen = GeneratorNSF(cfg).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    gen.load_state_dict(state["generator"])
    gen.remove_weight_norm()
    gen.eval()

    # 1 s of features @ 100 Hz with 1024-dim WavLM features
    T = 100
    x = torch.randn(1, T, cfg.hubert_dim, device=device) * 0.1
    f0 = torch.full((1, T), 220.0, device=device)  # A3
    pitch = torch.full((1, T), 100, dtype=torch.long, device=device)

    with torch.inference_mode():
        wav = gen(x, f0, pitch)
    wav_np = wav.squeeze().cpu().numpy()
    print(f"output: shape={wav_np.shape} dtype={wav_np.dtype} "
          f"min={wav_np.min():.3f} max={wav_np.max():.3f} "
          f"rms={float(np.sqrt(np.mean(wav_np ** 2))):.4f}")
    expected_len = T * int(np.prod(cfg.upsample_rates))
    if wav_np.shape[-1] != expected_len:
        print(f"FAIL: expected length {expected_len}, got {wav_np.shape[-1]}")
        return 2
    if not np.isfinite(wav_np).all():
        print("FAIL: non-finite values in output")
        return 3
    if float(np.abs(wav_np).max()) < 1e-4:
        print("FAIL: output is silence")
        return 4

    out_path = ROOT / "outputs" / "smoke.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(out_path, wav_np, int(cfg.sampling_rate))
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
