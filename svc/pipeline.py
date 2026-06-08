"""Top-level zero-shot SVC inference pipeline.

Wires together: WavLM encoder, F0 extraction, kNN retrieval, NSF vocoder.
The whole conversion API is `convert(...)`. The CLI and Gradio app call this
and nothing else, so the two interfaces never drift.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

from .content import WavLMEncoder
from .match import knn_match
from .pitch import (
    autotune_f0,
    coarse_f0,
    compute_f0,
    compute_pitch_shift_factor,
    median_filter_f0,
    semitone_factor,
)
from .synth import GeneratorNSF
from .utils.audio import extract_voiced_area, load_wav
from .utils.tools import AttrDict

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_SR = 24000
DEFAULT_VOICE_DIR = Path("voices")

ProgressFn = Callable[[float, str], None]


@dataclass
class ConversionConfig:
    topk: int = 4
    pitch_shift_semitones: float | None = None  # None => auto from ref median F0
    speech_enroll: bool = False
    alpha: float = 0.0
    target_loudness_db: float | None = -16.0
    vad_trim_reference: bool = True
    f0_method: str = "fcpe"  # "fcpe" (default, neural), "praat", "pyin", "median"
    f0_filter_radius: int = 3   # median filter radius for the F0 contour (1 = off)
    autotune: bool = False      # snap voiced F0 to nearest equal-temp semitone
    protect: float = 0.5        # 0..1, source weight blended into voiceless frames
    rms_mix_rate: float = 0.25  # 0..1, copy source RMS envelope onto output
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT_DIR)


class SVCPipeline:
    """Holds the loaded models so we don't pay the load cost on each conversion."""

    def __init__(self, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
                 device: str | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        wavlm_path = self.checkpoint_dir / "WavLM-Large.pt"
        synth_ckpt = self.checkpoint_dir / "G_150k.pt"
        synth_cfg = Path(__file__).parent / "synth" / "configs" / "g_150k.json"

        for p in (wavlm_path, synth_ckpt, synth_cfg):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required model file missing: {p}. Run scripts/download_models.py"
                )

        log.info("loading WavLM-Large from %s", wavlm_path)
        self.encoder = WavLMEncoder(wavlm_path, device=self.device)

        log.info("loading NSF synth from %s", synth_ckpt)
        with open(synth_cfg) as f:
            cfg = AttrDict(json.load(f))
        gen = GeneratorNSF(cfg).to(self.device)
        state = torch.load(synth_ckpt, map_location="cpu", weights_only=False)
        gen.load_state_dict(state["generator"])
        gen.remove_weight_norm()
        self.synth = gen.eval()
        self.h = cfg
        self.sample_rate = int(cfg.sampling_rate)
        # ref-feature cache. key = (path, mtime_ns, size, vad_trim_flag)
        # huge speedup when the user re-converts with the same reference
        # (Gradio UI, batch jobs, etc).
        self._ref_cache: dict[tuple, Tensor] = {}
        self.voice_dir = DEFAULT_VOICE_DIR
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        (self.voice_dir / "cache").mkdir(parents=True, exist_ok=True)

    def clear_reference_cache(self) -> None:
        self._ref_cache.clear()

    def _ref_disk_key(self, paths: list[Path], do_vad_trim: bool) -> str:
        import hashlib
        h = hashlib.sha1()
        for p in paths:
            try:
                st = p.stat()
                h.update(str(p.resolve()).encode("utf-8"))
                h.update(str(st.st_mtime_ns).encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
            except OSError:
                h.update(str(p.resolve()).encode("utf-8"))
        h.update(b"vad" if do_vad_trim else b"raw")
        return h.hexdigest()[:16]

    def save_voice_profile(self, name: str, reference_paths: list[str | Path],
                           vad_trim: bool = True) -> Path:
        """Encode the given reference(s) and save as voices/<name>.npz so they
        can be reloaded in ~milliseconds instead of re-running WavLM."""
        paths = [Path(p) for p in reference_paths]
        wavs = _prepare_reference_wavs(paths, vad_trim)
        feats = self.encoder.encode_paths(wavs).to(self.device)
        out = self.voice_dir / f"{name}.npz"
        np.savez(out, feats=feats.cpu().numpy().astype(np.float32))
        log.info("saved voice profile %s (%d frames) -> %s",
                 name, feats.shape[0], out)
        return out

    def load_voice_profile(self, name: str) -> Tensor:
        """Load a previously saved voice profile by name."""
        p = self.voice_dir / f"{name}.npz"
        if not p.exists():
            raise FileNotFoundError(f"voice profile not found: {p}")
        data = np.load(p)
        feats = torch.from_numpy(data["feats"]).to(self.device)
        log.info("loaded voice profile %s (%d frames)", name, feats.shape[0])
        return feats

    def list_voice_profiles(self) -> list[str]:
        return sorted(p.stem for p in self.voice_dir.glob("*.npz")
                      if p.parent == self.voice_dir)

    def _cached_ref_features(self, paths: list[Path], do_vad_trim: bool,
                              on_chunk=None) -> Tensor:
        key_parts = []
        for p in paths:
            try:
                st = p.stat()
                key_parts.append((str(p.resolve()), st.st_mtime_ns, st.st_size))
            except OSError:
                key_parts.append((str(p.resolve()), 0, 0))
        key = (tuple(key_parts), bool(do_vad_trim))
        hit = self._ref_cache.get(key)
        if hit is not None:
            log.info("reference cache HIT (memory): %d frames", hit.shape[0])
            return hit
        # disk cache: hash of (path, mtime, size, vad-flag) -> npz
        disk_key = self._ref_disk_key(paths, do_vad_trim)
        disk_path = self.voice_dir / "cache" / f"{disk_key}.npz"
        if disk_path.exists():
            try:
                feats = torch.from_numpy(np.load(disk_path)["feats"]).to(self.device)
                log.info("reference cache HIT (disk): %d frames from %s",
                         feats.shape[0], disk_path.name)
                if len(self._ref_cache) >= 8:
                    self._ref_cache.pop(next(iter(self._ref_cache)))
                self._ref_cache[key] = feats
                return feats
            except Exception as e:
                log.warning("disk cache read failed (%s); recomputing", e)

        ref_wavs_for_encoder = _prepare_reference_wavs(paths, do_vad_trim)
        feats = self.encoder.encode_paths(ref_wavs_for_encoder,
                                          on_chunk=on_chunk).to(self.device)
        try:
            np.savez(disk_path, feats=feats.cpu().numpy().astype(np.float32))
        except Exception as e:
            log.warning("disk cache write failed: %s", e)
        # cap cache size to keep VRAM bounded (each entry can be ~tens of MB)
        if len(self._ref_cache) >= 8:
            self._ref_cache.pop(next(iter(self._ref_cache)))
        self._ref_cache[key] = feats
        return feats

    @torch.inference_mode()
    def convert(self, source_path: str | Path,
                reference_paths: list[str | Path] | str | Path,
                cfg: ConversionConfig | None = None,
                progress: ProgressFn | None = None) -> tuple[np.ndarray, int]:
        cfg = cfg or ConversionConfig()
        if isinstance(reference_paths, (str, Path)):
            reference_paths = [reference_paths]

        source_path = Path(source_path)
        reference_paths = [Path(p) for p in reference_paths]

        # stage weights (sum=1.0): source_enc, ref_enc, f0, knn, vocode, finalize
        stages = {
            "src": (0.00, 0.12),
            "ref": (0.12, 0.35),
            "f0":  (0.35, 0.60),
            "knn": (0.60, 0.72),
            "voc": (0.72, 0.97),
            "fin": (0.97, 1.00),
        }

        def _emit(stage: str, sub_frac: float, desc: str) -> None:
            lo, hi = stages[stage]
            frac = lo + max(0.0, min(1.0, sub_frac)) * (hi - lo)
            log.info("[%3d%%] %s", int(frac * 100), desc)
            if progress is not None:
                try:
                    progress(frac, desc=desc)
                except TypeError:
                    progress(frac, desc)

        # 1) WavLM features for source
        _emit("src", 0.0, "encoding source (WavLM)")
        t = time.perf_counter()
        src_chunk_cb = lambda d, n: _emit(  # noqa: E731
            "src", d / max(n, 1),
            f"encoding source: chunk {d}/{n}",
        )
        query_seq = self.encoder.encode_path(source_path, on_chunk=src_chunk_cb).to(self.device)
        log.debug("source encode: %.2fs -> %d frames", time.perf_counter() - t, query_seq.shape[0])

        # 2) Build matching pool. cached per (path, mtime, vad-flag).
        _emit("ref", 0.0, f"encoding {len(reference_paths)} reference file(s)")
        ref_chunk_cb = lambda d, n: _emit(  # noqa: E731
            "ref", d / max(n, 1),
            f"encoding reference: slice {d}/{n}",
        )
        synth_set = self._cached_ref_features(reference_paths, cfg.vad_trim_reference,
                                              on_chunk=ref_chunk_cb)
        _emit("ref", 1.0,
              f"matching pool: {synth_set.shape[0]} frames "
              f"({synth_set.shape[0]/50.0:.1f}s @ 50Hz)")

        # 3) F0 + voicing on the source.
        _emit("f0", 0.0, f"computing F0 ({cfg.f0_method})")
        t = time.perf_counter()
        f0_src = compute_f0(str(source_path), method=cfg.f0_method)
        _emit("f0", 0.4, f"F0 done in {time.perf_counter()-t:.2f}s")
        if cfg.pitch_shift_semitones is not None:
            factor = semitone_factor(cfg.pitch_shift_semitones)
        else:
            factor = compute_pitch_shift_factor(
                f0_src, str(reference_paths[0]), speech_enroll=cfg.speech_enroll,
            )
        # report the snapped semitone shift so it's obvious what auto chose
        inferred_st = 12.0 * float(np.log2(factor)) if factor > 0 else 0.0
        _emit("f0", 0.6,
              f"F0 shift factor: {factor:.3f} ({inferred_st:+.1f} st)")
        f0_src = f0_src * factor
        # smooth out octave-jump outliers in the F0 contour
        if cfg.f0_filter_radius and cfg.f0_filter_radius > 1:
            f0_src = median_filter_f0(f0_src, radius=int(cfg.f0_filter_radius))
            _emit("f0", 0.8,
                  f"F0 median filter r={int(cfg.f0_filter_radius)}")
        if cfg.autotune:
            f0_src = autotune_f0(f0_src)
            _emit("f0", 0.9, "F0 autotune (snap to nearest semitone)")
        _emit("f0", 1.0, "F0 ready")
        pitch_src = coarse_f0(f0_src, f0_bins=int(self.h.f0_bins))

        query_mask_np = extract_voiced_area(str(source_path), hop_size=480, energy_thres=0.1)
        query_mask = torch.from_numpy(query_mask_np).to(self.device)

        # 4) kNN retrieval
        _emit("knn", 0.0, f"kNN match topk={cfg.topk} alpha={cfg.alpha:.2f}")
        knn_cb = lambda d, n: _emit(  # noqa: E731
            "knn", d / max(n, 1),
            f"kNN: chunk {d}/{n}",
        )
        matched = knn_match(query_seq, synth_set, topk=cfg.topk, alpha=cfg.alpha,
                            device=self.device, on_chunk=knn_cb)

        # 5) Align voiced mask + F0 length to feature length, then upsample
        # features 2x (50 -> 100 Hz) to match what the vocoder expects.
        query_len = matched.shape[0]
        if len(query_mask) > query_len:
            query_mask = query_mask[:query_len]
        elif len(query_mask) < query_len:
            pad = query_len - len(query_mask)
            query_mask = torch.cat([query_mask,
                                    torch.zeros(pad, dtype=query_mask.dtype,
                                                device=query_mask.device)])
        # blend matched (target) and source features per frame. on voiced
        # frames we always lean fully on matched (target voice), on voiceless
        # frames `protect` controls how much raw source we mix back in to keep
        # consonants crisp without leaking source timbre across the whole song.
        # protect=0 -> pure matched everywhere (may sound 'alien' on consonants).
        # protect=1 -> raw source on voiceless (current behaviour, safest, can
        # leak source).
        protect = float(max(0.0, min(1.0, cfg.protect)))
        mask_v = query_mask.float()[..., None]              # 1.0 voiced, 0 unvoiced
        unvoiced_w = (1.0 - mask_v) * protect              # source weight on unvoiced
        matched_w = mask_v + (1.0 - mask_v) * (1.0 - protect)  # matched weight
        out_feats = matched * matched_w + query_seq * unvoiced_w

        f0_len = query_len * 2
        f0_src = _align_length(f0_src, f0_len)
        pitch_src = _align_length(pitch_src, f0_len)

        out_feats = torch.repeat_interleave(out_feats, 2, dim=0)  # 50Hz -> 100Hz

        # 6) Vocode. Chunked so long songs get progress and don't blow VRAM.
        # NSF HiFi-GAN is fully convolutional so chunking with small overlap +
        # equal-power crossfade is inaudible.
        n_feat = out_feats.shape[0]
        _emit("voc", 0.0, f"vocoding {n_feat} feature frames")
        f0_t = torch.from_numpy(f0_src).float().to(self.device)
        pitch_t = torch.from_numpy(pitch_src).to(self.device)

        chunk_frames = 1000  # ~10s of output per chunk at 100Hz feature rate
        overlap = 50         # ~0.5s overlap for crossfade
        if n_feat <= chunk_frames + overlap:
            wav = self.synth(out_feats[None], f0_t.unsqueeze(0), pitch_t.unsqueeze(0))
            wav = wav.squeeze(0).squeeze(0).cpu()
            _emit("voc", 1.0, "vocoding done")
        else:
            wav = self._vocode_chunked(out_feats, f0_t, pitch_t,
                                       chunk_frames=chunk_frames, overlap=overlap,
                                       emit=lambda f, d: _emit("voc", f, d))

        # free the big intermediates before loudness/clip so empty_cache below
        # can actually hand memory back to the driver
        del query_seq, matched, out_feats, f0_t, pitch_t, query_mask

        # 7) Loudness match
        _emit("fin", 0.2, "loudness match")
        if cfg.target_loudness_db is not None:
            try:
                src_loud = torchaudio.functional.loudness(wav[None], self.sample_rate)
                wav = torchaudio.functional.gain(wav, float(cfg.target_loudness_db - src_loud))
            except RuntimeError:
                # torchaudio loudness needs >= ~400ms audio; for shorter clips skip
                log.debug("skipping loudness match for short clip")

        # 8) Soft clip protection
        wav_np = wav.numpy()
        peak = float(np.abs(wav_np).max())
        if peak > 0.98:
            wav_np = wav_np * (0.98 / peak)

        # 9) RMS envelope match: copy source loudness contour onto output so it
        # 'breathes' with the original performance instead of being flatlined.
        if cfg.rms_mix_rate and cfg.rms_mix_rate > 0:
            try:
                wav_np = _apply_rms_envelope(
                    wav_np, str(source_path), self.sample_rate,
                    rate=float(cfg.rms_mix_rate),
                )
                _emit("fin", 0.8, f"RMS envelope match rate={cfg.rms_mix_rate:.2f}")
            except Exception as e:
                log.warning("RMS envelope match failed: %s", e)

        # release the cuda caching allocator's reserved pool back to the driver.
        # without this Task Manager / nvidia-smi keep showing peak VRAM forever
        # even though torch internally knows the memory is free.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _emit("fin", 1.0, "done")

        return wav_np.astype(np.float32), self.sample_rate

    def _vocode_chunked(self, out_feats: Tensor, f0_t: Tensor, pitch_t: Tensor,
                        *, chunk_frames: int, overlap: int,
                        emit) -> Tensor:
        """Run the NSF vocoder in chunks with equal-power crossfade between
        adjacent chunks. Lets us show progress on long songs without breaking
        audio at the seams."""
        n_feat = out_feats.shape[0]
        # feature frames -> audio samples ratio (vocoder upsample product)
        upp = int(self.synth.upp)
        sr_per_frame = upp  # NSF upsamples each feature frame to `upp` samples
        starts = list(range(0, n_feat, chunk_frames))
        n_chunks = len(starts)
        wav_chunks: list[Tensor] = []
        prev_tail: Tensor | None = None  # in audio-sample domain
        xfade_samples = overlap * sr_per_frame
        fade_in = torch.linspace(0.0, 1.0, xfade_samples)
        fade_out = 1.0 - fade_in

        for i, start in enumerate(starts):
            end = min(start + chunk_frames + overlap, n_feat)
            f_chunk = out_feats[start:end]
            # out_feats is at 100Hz, f0_t/pitch_t also at 100Hz, so frame index
            # is direct.
            f0_chunk = f0_t[start:end]
            pitch_chunk = pitch_t[start:end]

            seg = self.synth(f_chunk[None], f0_chunk.unsqueeze(0),
                             pitch_chunk.unsqueeze(0))
            seg = seg.squeeze(0).squeeze(0).cpu()

            if prev_tail is None:
                wav_chunks.append(seg)
            else:
                # crossfade prev tail with current head
                head_len = min(xfade_samples, seg.shape[0], prev_tail.shape[0])
                head = seg[:head_len]
                tail = prev_tail[-head_len:]
                fi = fade_in[:head_len]
                fo = fade_out[:head_len]
                blended = tail * fo + head * fi
                # replace last `head_len` of previous chunk with blended, then
                # append the rest of this chunk after the head
                wav_chunks[-1] = torch.cat([wav_chunks[-1][:-head_len], blended])
                wav_chunks.append(seg[head_len:])

            # stash a tail copy for next iter crossfade
            if end < n_feat:
                prev_tail = seg[-xfade_samples:].clone()
            else:
                prev_tail = None

            emit((i + 1) / n_chunks, f"vocoding: chunk {i+1}/{n_chunks}")

        return torch.cat(wav_chunks, dim=0)

    def vram_stats(self) -> dict:
        """Returns dict of MiB values: {allocated, reserved, peak_allocated, total}."""
        if self.device.type != "cuda":
            return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "total": 0.0}
        mb = 1024 * 1024
        # mem_get_info needs an indexed device; self.device may be plain "cuda"
        idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        return {
            "allocated": torch.cuda.memory_allocated(idx) / mb,
            "reserved": torch.cuda.memory_reserved(idx) / mb,
            "peak_allocated": torch.cuda.max_memory_allocated(idx) / mb,
            "total": total / mb,
            "free": free / mb,
        }

    def release_vram(self, clear_ref_cache: bool = False) -> None:
        """Drop reference cache (optional) and hand reserved VRAM back to driver."""
        if clear_ref_cache:
            self.clear_reference_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)


def _prepare_reference_wavs(paths: list[Path], do_vad_trim: bool) -> list[Path | Tensor]:
    """Returns a list usable by `WavLMEncoder.encode_paths`. If VAD trim is
    enabled, we materialise pre-trimmed waveforms into in-memory paths via a
    temp dir... actually simpler: we let the encoder load raw paths when VAD
    is disabled, and pass pre-loaded tensors via a thin shim when enabled."""
    if not do_vad_trim:
        return list(paths)

    from preprocess.vad import trim_silence_file

    trimmed_paths: list[Path] = []
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="svc_ref_vad_"))
    for i, p in enumerate(paths):
        trimmed = trim_silence_file(p, sr=16000)
        if trimmed.size == 0:
            log.warning("VAD trim produced empty audio for %s; using raw file", p)
            trimmed_paths.append(p)
            continue
        out_p = tmp_dir / f"ref_{i}.wav"
        sf.write(str(out_p), trimmed, 16000)
        trimmed_paths.append(out_p)
    return trimmed_paths


def _align_length(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) > target:
        return arr[:target]
    if len(arr) < target:
        pad = target - len(arr)
        return np.pad(arr, (0, pad), mode="edge")
    return arr


def _apply_rms_envelope(out_wav: np.ndarray, source_path: str, sr: int,
                        rate: float, frame_length: int = 2048,
                        hop_length: int = 256) -> np.ndarray:
    """Borrow the source's per-frame RMS envelope onto the output waveform.
    `rate` in [0, 1] sets how strongly to match (0 = no change, 1 = exact match)
    via a per-sample gain `(src_rms / out_rms) ** rate`. Smooths the gain curve
    by linear interp from the frame grid so we don't get zipper noise."""
    import librosa

    src, src_sr = sf.read(source_path, always_2d=False)
    if src.ndim > 1:
        src = src.mean(axis=1)
    if src_sr != sr:
        src = librosa.resample(src.astype(np.float32), orig_sr=src_sr, target_sr=sr)
    n = min(len(src), len(out_wav))
    src = src[:n].astype(np.float32)
    out = out_wav[:n].astype(np.float32).copy()

    src_rms = librosa.feature.rms(y=src, frame_length=frame_length,
                                  hop_length=hop_length).flatten()
    out_rms = librosa.feature.rms(y=out, frame_length=frame_length,
                                  hop_length=hop_length).flatten()
    eps = 1e-6
    ratio = (src_rms + eps) / (out_rms + eps)
    ratio = ratio ** float(rate)

    frame_centers = (np.arange(len(ratio)) + 0.5) * hop_length
    sample_idx = np.arange(n)
    gain = np.interp(sample_idx, frame_centers, ratio).astype(np.float32)
    out = out * gain
    # protect against any rare clipping introduced by envelope copy
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    if len(out_wav) > n:
        # tail beyond source length: leave untouched
        out = np.concatenate([out, out_wav[n:]])
    return out


def convert(source_path: str | Path,
            reference_paths: list[str | Path] | str | Path,
            *,
            out_path: str | Path | None = None,
            checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
            device: str | None = None,
            **cfg_kwargs) -> tuple[np.ndarray, int]:
    """One-shot convenience entry point. Loads the models, runs a conversion,
    optionally writes the wav. Use `SVCPipeline` directly if you want to amortise
    model load across many conversions."""
    pipe = SVCPipeline(checkpoint_dir=checkpoint_dir, device=device)
    cfg = ConversionConfig(**cfg_kwargs) if cfg_kwargs else ConversionConfig()
    wav, sr = pipe.convert(source_path, reference_paths, cfg=cfg)
    if out_path is not None:
        sf.write(str(out_path), wav, sr)
    return wav, sr
