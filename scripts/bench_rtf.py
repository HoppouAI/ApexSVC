"""Measure RTF (Real-Time Factor) for SVCPipeline.convert.

RTF = process_seconds / audio_seconds. Lower is better. <1.0 = faster than realtime.

Runs:
  1. warm-up convert (load CUDA kernels, jit caches)
  2. timed runs at several source durations with a fixed 30s reference
  3. timed runs varying reference length at a fixed source

Reports per-stage timing too (encode src, encode ref, F0, kNN, vocode).
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


def make_voice(dur_s: float, base_hz: float = 200.0) -> np.ndarray:
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    f0 = base_hz + 30.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.18, 0.1], start=1):
        sig += amp * np.sin(k * phase)
    env = np.minimum(1.0, np.minimum(t / 0.05, (dur_s - t) / 0.05))
    sig *= env
    return (0.6 * sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def write(path: Path, dur: float, base_hz: float = 200.0) -> Path:
    sf.write(path, make_voice(dur, base_hz), SR)
    return path


def time_convert(pipe, src_path, ref_paths, cfg, label: str) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    wav, sr = pipe.convert(src_path, ref_paths, cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    audio_s = wav.shape[0] / sr
    rtf = dt / audio_s
    print(f"  [{label}] audio={audio_s:5.2f}s  proc={dt:6.2f}s  RTF={rtf:.3f}  "
          f"({1.0 / rtf:5.1f}x realtime)")
    return rtf


def main() -> int:
    bench_dir = ROOT / "outputs" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("loading pipeline ...")
    t0 = time.perf_counter()
    pipe = SVCPipeline()
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s  "
          f"(device={pipe.device})")

    cfg = ConversionConfig(topk=4, alpha=0.0, vad_trim_reference=False,
                           speech_enroll=False)

    # baseline reference (30 s) used for the src-length sweep
    ref_30 = write(bench_dir / "ref_30s.wav", 30.0, base_hz=210.0)

    # warm-up: small convert to trigger cuda compile, weight load is already done
    print("\nwarm-up:")
    src_warm = write(bench_dir / "src_warm.wav", 2.0, base_hz=180.0)
    time_convert(pipe, src_warm, [ref_30], cfg, "warm")
    time_convert(pipe, src_warm, [ref_30], cfg, "warm2")

    print("\nsource-length sweep (ref = 30 s):")
    src_durs = [3.0, 10.0, 30.0, 60.0]
    src_rtfs = []
    for d in src_durs:
        p = write(bench_dir / f"src_{int(d)}s.wav", d, base_hz=185.0)
        src_rtfs.append((d, time_convert(pipe, p, [ref_30], cfg, f"src{int(d):>3}s")))

    print("\nreference-length sweep (src = 10 s):")
    src_10 = write(bench_dir / "src_10s_b.wav", 10.0, base_hz=195.0)
    ref_durs = [10.0, 30.0, 60.0, 180.0, 300.0]
    ref_rtfs = []
    for d in ref_durs:
        rp = write(bench_dir / f"ref_{int(d)}s.wav", d, base_hz=215.0)
        ref_rtfs.append((d, time_convert(pipe, src_10, [rp], cfg, f"ref{int(d):>3}s")))

    print("\nsummary:")
    print("  source len -> RTF (lower is better)")
    for d, r in src_rtfs:
        print(f"    {d:6.1f}s : {r:.3f}  ({1.0 / r:5.1f}x realtime)")
    print("  reference len -> RTF (with 10 s source)")
    for d, r in ref_rtfs:
        print(f"    {d:6.1f}s : {r:.3f}  ({1.0 / r:5.1f}x realtime)")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\nCUDA peak alloc: {peak_gb:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
