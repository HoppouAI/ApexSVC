"""End-to-end identity test on synthetic speech-like audio.

Generates a short FM-modulated harmonic tone (vaguely vocal),
runs SVCPipeline.convert with that file as both source and reference,
checks the output: correct sr, non-trivial RMS, reasonable F0 retention.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svc.pipeline import SVCPipeline, ConversionConfig  # noqa: E402


def make_synth_voice(sr: int = 24000, dur: float = 3.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Pitch contour 180 -> 230 -> 200 Hz
    f0 = 180.0 + 25.0 * np.sin(2 * np.pi * 0.5 * t) + 25.0
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros_like(t)
    # harmonics with falling amplitude (vaguely vowel-like)
    for k, amp in enumerate([1.0, 0.55, 0.35, 0.2, 0.12, 0.07], start=1):
        sig += amp * np.sin(k * phase)
    # gentle envelope
    env = np.minimum(1.0, np.minimum(t / 0.1, (dur - t) / 0.1))
    sig *= env
    sig = 0.6 * sig / (np.abs(sig).max() + 1e-9)
    return sig.astype(np.float32)


def main() -> int:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    src_path = out_dir / "identity_src.wav"
    ref_path = out_dir / "identity_ref.wav"
    sig = make_synth_voice()
    sf.write(src_path, sig, 24000)
    sf.write(ref_path, sig, 24000)

    print("loading pipeline (may take a few sec) ...")
    t0 = time.time()
    pipe = SVCPipeline()
    print(f"loaded in {time.time() - t0:.1f}s")

    cfg = ConversionConfig(topk=4, alpha=0.0, vad_trim_reference=False,
                           speech_enroll=False)
    print("converting (identity) ...")
    t1 = time.time()
    out, sr = pipe.convert(src_path, [ref_path], cfg)
    dt = time.time() - t1
    print(f"converted in {dt:.2f}s -> shape={out.shape}, sr={sr}")

    out_path = out_dir / "identity_out.wav"
    sf.write(out_path, out, sr)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    peak = float(np.abs(out).max())
    finite = bool(np.isfinite(out).all())
    print(f"out rms={rms:.4f} peak={peak:.3f} finite={finite}")

    fail = []
    if not finite:
        fail.append("non-finite output")
    if rms < 1e-3:
        fail.append(f"output too quiet (rms={rms})")
    if peak > 1.001:
        fail.append(f"output clipped (peak={peak})")
    if out.shape[0] < int(0.5 * sr):
        fail.append(f"output too short ({out.shape[0]} samples)")

    if fail:
        print("FAIL:")
        for f in fail:
            print("  -", f)
        return 1
    print(f"OK: identity convert wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
