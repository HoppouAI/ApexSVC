"""Benchmark repeat conversions with the SAME reference, to measure cache hit.

Pattern: load pipeline, convert(src_A, ref_X), then convert(src_B, ref_X).
Expect: second convert is MUCH faster because reference features are cached.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svc.pipeline import SVCPipeline, ConversionConfig  # noqa: E402

SR = 24000


def voice(dur, base_hz=200.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f0 = base_hz + 30.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(amp * np.sin(k * phase)
              for k, amp in enumerate([1.0, 0.5, 0.3, 0.18, 0.1], 1))
    env = np.minimum(1.0, np.minimum(t / 0.05, (dur - t) / 0.05))
    return (0.6 * sig * env / (np.abs(sig * env).max() + 1e-9)).astype(np.float32)


def t_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    out = ROOT / "outputs" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "src_a.wav", voice(10.0, 195.0), SR)
    sf.write(out / "src_b.wav", voice(10.0, 175.0), SR)
    sf.write(out / "ref_long.wav", voice(180.0, 215.0), SR)  # 3 min ref

    pipe = SVCPipeline()
    cfg = ConversionConfig(vad_trim_reference=False, speech_enroll=False)

    # warmup with a small ref
    sf.write(out / "ref_warm.wav", voice(10.0), SR)
    pipe.convert(out / "src_a.wav", [out / "ref_warm.wav"], cfg)

    print("\nrepeat-ref cache benchmark (10s src, 180s ref):")
    t0 = t_sync()
    pipe.convert(out / "src_a.wav", [out / "ref_long.wav"], cfg)
    dt_cold = t_sync() - t0
    print(f"  first  (cache MISS): {dt_cold:.2f}s")

    t0 = t_sync()
    pipe.convert(out / "src_b.wav", [out / "ref_long.wav"], cfg)
    dt_warm = t_sync() - t0
    print(f"  second (cache HIT) : {dt_warm:.2f}s   speedup: {dt_cold / dt_warm:.1f}x")

    t0 = t_sync()
    pipe.convert(out / "src_a.wav", [out / "ref_long.wav"], cfg)
    dt_warm2 = t_sync() - t0
    print(f"  third  (cache HIT) : {dt_warm2:.2f}s   speedup: {dt_cold / dt_warm2:.1f}x")


if __name__ == "__main__":
    main()
