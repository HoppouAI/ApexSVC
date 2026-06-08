"""Profile each stage of SVCPipeline.convert on a 10s source + 30s ref.

Times only the inner stages so we can attack the biggest one.
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
from svc.pitch import compute_f0, coarse_f0, compute_pitch_shift_factor  # noqa: E402
from svc.match import knn_match  # noqa: E402
from svc.utils.audio import extract_voiced_area  # noqa: E402

SR = 24000


def voice(dur, base_hz=200.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f0 = base_hz + 30.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(amp * np.sin(k * phase)
              for k, amp in enumerate([1.0, 0.5, 0.3, 0.18, 0.1], 1))
    env = np.minimum(1.0, np.minimum(t / 0.05, (dur - t) / 0.05))
    sig = sig * env
    return (0.6 * sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def t_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    out_dir = ROOT / "outputs" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_p = out_dir / "src.wav"; sf.write(src_p, voice(10.0, 195.0), SR)
    ref_p = out_dir / "ref.wav"; sf.write(ref_p, voice(30.0, 215.0), SR)

    print("loading pipeline ...")
    pipe = SVCPipeline()
    cfg = ConversionConfig(vad_trim_reference=False, speech_enroll=False)

    # warmup
    pipe.convert(src_p, [ref_p], cfg)

    print("\n-- stage timings (10s src / 30s ref) --")
    timings = {}

    t0 = t_sync()
    query_seq = pipe.encoder.encode_path(src_p).to(pipe.device)
    timings["1. encode source (WavLM)"] = t_sync() - t0

    t0 = t_sync()
    synth_set = pipe.encoder.encode_paths([ref_p]).to(pipe.device)
    timings["2. encode reference (WavLM)"] = t_sync() - t0

    t0 = t_sync()
    f0_src = compute_f0(str(src_p))
    timings["3. F0 source (PYIN + parselmouth)"] = t_sync() - t0

    t0 = t_sync()
    factor = compute_pitch_shift_factor(f0_src, str(ref_p), speech_enroll=False)
    timings["4. ref mean-F0 for pitch shift"] = t_sync() - t0

    t0 = t_sync()
    f0_src_shifted = f0_src * factor
    pitch_src = coarse_f0(f0_src_shifted, f0_bins=int(pipe.h.f0_bins))
    timings["5. coarse_f0 bucket"] = t_sync() - t0

    t0 = t_sync()
    mask = extract_voiced_area(str(src_p), hop_size=480, energy_thres=0.1)
    timings["6. voiced mask (STFT)"] = t_sync() - t0

    t0 = t_sync()
    matched = knn_match(query_seq, synth_set, topk=cfg.topk, alpha=cfg.alpha,
                        device=pipe.device)
    timings["7. kNN match"] = t_sync() - t0

    query_len = matched.shape[0]
    qmask = torch.from_numpy(mask).to(pipe.device)
    if len(qmask) > query_len:
        qmask = qmask[:query_len]
    elif len(qmask) < query_len:
        qmask = torch.cat([qmask, torch.zeros(query_len - len(qmask),
                                              dtype=qmask.dtype, device=qmask.device)])
    mb = qmask[..., None].repeat([1, matched.shape[-1]])
    out_feats = matched * mb + query_seq * (~mb.bool())
    out_feats = torch.repeat_interleave(out_feats, 2, dim=0)

    f0_len = query_len * 2
    def _align(a, n):
        if len(a) > n: return a[:n]
        if len(a) < n: return np.pad(a, (0, n - len(a)), mode="edge")
        return a
    f0_t = torch.from_numpy(_align(f0_src_shifted, f0_len)).float().to(pipe.device)
    pitch_t = torch.from_numpy(_align(pitch_src, f0_len)).to(pipe.device)

    t0 = t_sync()
    with torch.inference_mode():
        wav = pipe.synth(out_feats[None], f0_t.unsqueeze(0), pitch_t.unsqueeze(0))
    timings["8. vocode (NSF)"] = t_sync() - t0

    t0 = t_sync()
    _ = wav.squeeze(0).squeeze(0).cpu().numpy()
    timings["9. GPU->CPU + numpy"] = t_sync() - t0

    total = sum(timings.values())
    print(f"\n{'stage':<40} {'sec':>8} {'pct':>7}")
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"{k:<40} {v:8.3f} {100 * v / total:6.1f}%")
    print(f"{'TOTAL':<40} {total:8.3f}")


if __name__ == "__main__":
    main()
