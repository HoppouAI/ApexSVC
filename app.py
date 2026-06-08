"""Gradio UI for SVC-Experiments.

Wraps `SVCPipeline` so the UI and CLI share the same converter. Adds tabs
for a step by step guide, settings tips, and troubleshooting.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import gradio as gr
import soundfile as sf

from svc.pipeline import DEFAULT_CHECKPOINT_DIR, ConversionConfig, SVCPipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("svc.app")

_PIPE: SVCPipeline | None = None

INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


F0_METHODS = ["fcpe", "praat", "pyin", "median"]
F0_HELP = {
    "fcpe":   "Neural pitch tracker (RMVPE family). Best on singing, recommended default.",
    "praat":  "Classical autocorrelation (parselmouth). Very fast, can octave-jump on vibrato.",
    "pyin":   "Probabilistic YIN (librosa). Accurate but slow, mostly for reference.",
    "median": "Median of pyin and praat. Robust but slow.",
}


def get_pipe() -> SVCPipeline:
    global _PIPE
    if _PIPE is None:
        log.info("Loading SVC pipeline (one time, ~5s)")
        _PIPE = SVCPipeline(checkpoint_dir=DEFAULT_CHECKPOINT_DIR)
    return _PIPE


def _audio_seconds(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _ref_summary(files: list[str] | None) -> str:
    if not files:
        return "_no reference files loaded_"
    total = sum(_audio_seconds(f) for f in files)
    if total < 30:
        warn = "  \n_tip: aim for at least 30s, ideally 1 to 5 min of clean voice for best results_"
    elif total > 600:
        warn = "  \n_tip: anything over ~5 min mostly just slows things down, the matching pool is already saturated_"
    else:
        warn = ""
    return f"**{len(files)} file(s)**, total **{total:.1f}s** (~{total/60:.2f} min){warn}"


def _src_summary(path: str | None) -> str:
    if not path:
        return "_no source loaded_"
    secs = _audio_seconds(path)
    return f"**{secs:.2f}s** source, ~{secs/60:.2f} min"


def _pitch_visibility(auto: bool):
    return gr.update(visible=not auto)


def _f0_help(method: str):
    return gr.update(info=F0_HELP.get(method, ""))


def _vram_md() -> str:
    if _PIPE is None:
        return "_pipeline not loaded yet (VRAM idle)_"
    s = _PIPE.vram_stats()
    if s["total"] == 0:
        return "_running on CPU, no VRAM tracking_"
    cached = len(_PIPE._ref_cache)
    return (
        f"**VRAM** allocated **{s['allocated']:.0f} MiB** | "
        f"reserved **{s['reserved']:.0f} MiB** | "
        f"peak **{s['peak_allocated']:.0f} MiB** | "
        f"free **{s['free']:.0f} / {s['total']:.0f} MiB**  \n"
        f"_reference cache: **{cached}** entry(ies)_"
    )


def _release_vram(clear_refs: bool):
    if _PIPE is None:
        return "_pipeline not loaded yet_"
    _PIPE.release_vram(clear_ref_cache=bool(clear_refs))
    return _vram_md() + "  \n_(released)_"


def _list_example_pairs() -> list[list]:
    """If inputs/ has files, surface them as Gradio examples."""
    if not INPUTS_DIR.exists():
        return []
    audio = sorted(p for p in INPUTS_DIR.glob("*")
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
    if len(audio) < 2:
        return []
    audio_sorted = sorted(audio, key=_audio_seconds)
    src = audio_sorted[0]
    ref = audio_sorted[-1]
    if src == ref:
        return []
    return [[str(src), [str(ref)]]]


def run_conversion(source_path, reference_files, f0_method, topk, auto_pitch,
                   pitch_shift, speech_enroll, alpha, vad_trim, loudness_db,
                   save_to_outputs, f0_filter_radius, autotune, protect,
                   rms_mix_rate, progress=gr.Progress()):
    if not source_path:
        raise gr.Error("Upload a source audio file first.")
    if not reference_files:
        raise gr.Error("Upload at least one reference audio file.")

    progress(0.05, desc="Loading models (cached after first run)")
    pipe = get_pipe()

    cfg = ConversionConfig(
        topk=int(topk),
        pitch_shift_semitones=None if auto_pitch else float(pitch_shift),
        speech_enroll=bool(speech_enroll),
        alpha=float(alpha),
        vad_trim_reference=bool(vad_trim),
        target_loudness_db=float(loudness_db) if loudness_db is not None else None,
        f0_method=str(f0_method),
        f0_filter_radius=int(f0_filter_radius),
        autotune=bool(autotune),
        protect=float(protect),
        rms_mix_rate=float(rms_mix_rate),
        device=str(pipe.device),
    )

    src_secs = _audio_seconds(source_path)
    ref_secs = sum(_audio_seconds(f) for f in reference_files)

    log.info("convert: src=%.1fs ref=%.1fs cfg=%s",
             src_secs, ref_secs, cfg)

    t0 = time.perf_counter()
    # the pipeline calls progress(frac, desc=...) at every sub-stage so the
    # gradio bar moves smoothly through encode source / encode ref / f0 / knn
    # / vocode (chunked) / finalize.
    wav, sr = pipe.convert(
        source_path, [Path(p) for p in reference_files], cfg=cfg,
        progress=progress,
    )
    elapsed = time.perf_counter() - t0
    rtf = elapsed / max(src_secs, 1e-6)
    xrt = (1.0 / rtf) if rtf > 0 else float("inf")

    if save_to_outputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"convert_{ts}.wav"
    else:
        out_path = Path(tempfile.mkdtemp(prefix="svc_out_")) / "converted.wav"
    sf.write(str(out_path), wav, sr)

    status = (
        f"**Done in {elapsed:.2f}s**  \n"
        f"RTF **{rtf:.3f}** ({xrt:.1f}x realtime)  \n"
        f"Output: `{out_path}` ({wav.shape[0] / sr:.2f}s @ {sr} Hz)  \n"
        f"Settings: f0=`{cfg.f0_method}` topk=`{cfg.topk}` alpha=`{cfg.alpha:.2f}` "
        f"protect=`{cfg.protect:.2f}` rms=`{cfg.rms_mix_rate:.2f}` "
        f"f0-filt=`{cfg.f0_filter_radius}`"
        f"{' autotune' if cfg.autotune else ''} "
        f"vad={'on' if cfg.vad_trim_reference else 'off'} "
        f"pitch={'auto' if cfg.pitch_shift_semitones is None else f'{cfg.pitch_shift_semitones:+.1f} st'}"
    )
    return str(out_path), status


CUSTOM_CSS = """
.svc-hero {
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(236,72,153,0.12));
    border: 1px solid rgba(120, 120, 160, 0.25);
}
.svc-hero h1 { margin: 0 0 4px 0; font-size: 1.8em; }
.svc-hero p  { margin: 0; opacity: 0.85; }
.svc-card .gr-block { border-radius: 12px; }
.svc-status {
    border-radius: 12px;
    padding: 12px 14px;
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(120,120,160,0.15);
    font-size: 0.95em;
}
footer { display: none !important; }
"""


GUIDE_MD = """
## Quick start

1. **Source**: upload the audio you want to convert (a vocal stem, a TTS clip, your own singing). The cleaner it is (no music, no reverb) the better. UVR / `bs_roformer` stems work great.
2. **Reference**: upload one or more clips of the target voice. **30 seconds to 5 minutes** is the sweet spot. More is fine but stops helping after a few minutes.
3. Pick an **F0 method** (FCPE is the recommended default).
4. Click **Convert**. First run loads models (~5s), subsequent runs reuse them and reuse the encoded reference if you don't change the file.

## What gets better with more reference

- **Timbre similarity**: more variety = more frames for the kNN matcher to draw from.
- **Phoneme coverage**: if the reference never says a sound your source contains, the matcher has to approximate it.
- **Robustness**: a tiny reference (under ~10s) tends to over-fit a single phrase.

## Speed expectations (RTX 3060)

| Source | Reference | Time | Realtime |
|---|---|---|---|
| 10s | 30s | ~0.9s | 11x |
| 30s | 30s | ~1.7s | 18x |
| 60s | 30s | ~2.9s | 21x |
| 3:25 song | 5 min ref | ~80s | 2.5x |

Repeat conversions with the **same reference file** are much faster because the encoded reference pool is cached.
"""


TIPS_MD = """
## Settings cheat sheet

### F0 method
- **fcpe** (default): neural, RMVPE family. Best for singing, handles vibrato and breathy frames cleanly.
- **praat**: classical, very fast, can octave-jump on hard vibrato.
- **pyin**: accurate but slow, mostly here for comparison.
- **median**: blends pyin and praat. Slowest, most conservative.

### topk
- **1 to 2**: sharper timbre, less smearing, can sound a bit "stitched" on phoneme boundaries.
- **3 to 5** (default 4): natural sounding, smooth transitions.
- **6 to 10**: very smooth, can lose target voice character.

### alpha
- **0.0** (default): pure retrieval, fully target voice.
- **0.1 to 0.3**: blends some source content back in. Keeps more of the original delivery and consonant articulation. Useful if the output sounds "alien" on certain phonemes.
- **0.5+**: closer to the source voice than the target. Only useful as a creative effect.

### Auto pitch / manual semitones
- Default auto pitch lines the source mean F0 up with the reference mean F0. Best for most cases.
- Turn off and use **manual semitones** if you want to convert a male source to a female reference without it sounding pitched up, or to keep a specific key.

### Speech reference checkbox
- Turn on if your reference is **spoken**, not sung. It bumps the auto pitch target by 1.2x to compensate for the fact that singing typically sits higher than speech.

### VAD trim reference
- Removes silence from the reference before building the matching pool. Almost always helpful, leave on.

### Loudness target
- Default -16 LUFS sits well with most mixes. Drop the slider very low to effectively skip loudness matching.

## Troubleshooting

| Symptom | Try |
|---|---|
| Buzzy or glitchy pitch on long held notes | switch F0 method to **fcpe** |
| "Alien voice" on specific words | raise **alpha** to 0.2 to 0.3 |
| Output sounds nothing like the reference | check reference quality, raise reference duration, lower **alpha** to 0.0 |
| Output is pitched too high or low | turn off **auto pitch**, use **manual semitones** |
| Cracks or harsh peaks | source might be clipping, the loudness step will normalize but cannot fix clipping |
| Silence handling weird | turn **VAD trim** on |
"""


ABOUT_MD = """
## About

Zero-shot singing voice conversion. Pipeline:

```
source.wav -> WavLM-Large layer 6 features -> kNN match against reference pool
                                                        |
                                                        v
                              source F0 (FCPE) -> NSF HiFi-GAN vocoder -> output.wav
```

Built on the NeuCoSVC2 architecture (WavLM content + kNN + NSF vocoder), with FCPE for neural pitch tracking.

- **Encoder**: WavLM-Large, 24 layers, 1024 dim, layer 6 features at 50 Hz.
- **Matcher**: cosine kNN, top-k mean.
- **F0**: FCPE (default), praat, pyin, or median.
- **Vocoder**: NSF HiFi-GAN, 24 kHz output.
- **VAD**: silero-vad for reference trimming.

The exact same `SVCPipeline.convert(...)` runs from the CLI, so anything you tune here you can lock in via the `svc convert ...` command.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SVC-Experiments") as demo:
        gr.HTML(
            """
            <div class="svc-hero">
                <h1>SVC-Experiments</h1>
                <p>Zero-shot singing voice conversion. WavLM + kNN retrieval + NSF vocoder, with FCPE neural pitch tracking.</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Convert"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes="svc-card"):
                        gr.Markdown("### 1. Source audio")
                        source = gr.Audio(
                            label="Source (will be converted)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        src_info = gr.Markdown("_no source loaded_")
                        source.change(_src_summary, inputs=source, outputs=src_info)

                        gr.Markdown("### 2. Reference voice")
                        refs = gr.File(
                            label="Reference (1 to N files, total 30s to 5min ideal)",
                            file_count="multiple",
                            file_types=["audio"],
                        )
                        ref_info = gr.Markdown("_no reference files loaded_")
                        refs.change(_ref_summary, inputs=refs, outputs=ref_info)

                        gr.Markdown("### 3. Convert")
                        go = gr.Button("Convert", variant="primary", size="lg")

                    with gr.Column(scale=5, elem_classes="svc-card"):
                        gr.Markdown("### Output")
                        out_audio = gr.Audio(
                            label="Converted audio",
                            type="filepath",
                            interactive=False,
                        )
                        status = gr.Markdown(
                            "_results will appear here after conversion_",
                            elem_classes="svc-status",
                        )

                        with gr.Accordion("Conversion settings", open=True):
                            f0_method = gr.Dropdown(
                                F0_METHODS, value="fcpe", label="F0 (pitch) method",
                                info=F0_HELP["fcpe"],
                            )
                            f0_method.change(
                                _f0_help, inputs=f0_method, outputs=f0_method,
                            )

                            with gr.Row():
                                topk = gr.Slider(
                                    1, 10, value=4, step=1, label="topk (kNN neighbours)",
                                    info="lower = sharper timbre, higher = smoother",
                                )
                                alpha = gr.Slider(
                                    0.0, 1.0, value=0.0, step=0.05, label="alpha (source mix)",
                                    info="0 = pure target, 0.2 to 0.3 fixes 'alien' words",
                                )

                            with gr.Row():
                                auto_pitch = gr.Checkbox(
                                    value=False, label="Auto pitch align",
                                    info="snaps to nearest whole semitone, folds to closest octave. leave off if source is already in a sensible range",
                                )
                                speech_enroll = gr.Checkbox(
                                    value=False, label="Reference is speech (not singing)",
                                    info="adds +3 semitones on top of auto",
                                )

                            pitch_shift = gr.Slider(
                                -12.0, 12.0, value=0.0, step=0.5,
                                label="Manual pitch shift (semitones)",
                                info="only used when auto pitch is off",
                                visible=False,
                            )
                            auto_pitch.change(
                                _pitch_visibility, inputs=auto_pitch, outputs=pitch_shift,
                            )

                            with gr.Row():
                                vad_trim = gr.Checkbox(
                                    value=True, label="Trim silence from reference",
                                    info="silero-vad based",
                                )
                                save_to_outputs = gr.Checkbox(
                                    value=True, label="Save into outputs/ folder",
                                    info="timestamped filename",
                                )

                            with gr.Accordion("Quality (Applio-style)", open=True):
                                with gr.Row():
                                    protect = gr.Slider(
                                        0.0, 1.0, value=0.5, step=0.05,
                                        label="Voiceless protect",
                                        info="source weight on consonants. 0 = pure target (may sound alien), 1 = full source (may leak)",
                                    )
                                    rms_mix_rate = gr.Slider(
                                        0.0, 1.0, value=0.25, step=0.05,
                                        label="RMS envelope match",
                                        info="copy source loudness contour. 0 = flat, 1 = exact source dynamics",
                                    )
                                with gr.Row():
                                    f0_filter_radius = gr.Slider(
                                        1, 7, value=3, step=2,
                                        label="F0 median filter radius",
                                        info="smooth pitch outliers. 1 = off, 3 mild, 5/7 stronger",
                                    )
                                    autotune = gr.Checkbox(
                                        value=False, label="Pitch autotune",
                                        info="snap each voiced frame to nearest semitone",
                                    )

                            loudness_db = gr.Slider(
                                -36.0, -6.0, value=-16.0, step=0.5,
                                label="Target loudness (LUFS)",
                                info="-16 is mix friendly; very short clips skip this automatically",
                            )

                        with gr.Accordion("VRAM", open=False):
                            vram_md = gr.Markdown(_vram_md())
                            with gr.Row():
                                refresh_vram = gr.Button("Refresh", size="sm")
                                free_vram = gr.Button("Free cached VRAM", size="sm")
                                free_all = gr.Button("Free VRAM + drop ref cache",
                                                     size="sm", variant="stop")
                            gr.Markdown(
                                "_Task Manager shows torch's reserved pool, not what's actively in use. "
                                "After a conversion the allocator keeps peak memory reserved for speed. "
                                "These buttons hand it back to the driver._"
                            )
                            refresh_vram.click(_vram_md, outputs=vram_md)
                            free_vram.click(lambda: _release_vram(False),
                                            outputs=vram_md)
                            free_all.click(lambda: _release_vram(True),
                                           outputs=vram_md)

                examples = _list_example_pairs()
                if examples:
                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=examples,
                        inputs=[source, refs],
                        label="files detected in inputs/",
                    )

            with gr.Tab("Guide"):
                gr.Markdown(GUIDE_MD)

            with gr.Tab("Tips & Troubleshooting"):
                gr.Markdown(TIPS_MD)

            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)

        go.click(
            run_conversion,
            inputs=[source, refs, f0_method, topk, auto_pitch, pitch_shift,
                    speech_enroll, alpha, vad_trim, loudness_db, save_to_outputs,
                    f0_filter_radius, autotune, protect, rms_mix_rate],
            outputs=[out_audio, status],
        ).then(_vram_md, outputs=vram_md)
    return demo


if __name__ == "__main__":
    ui = build_ui()
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink", neutral_hue="slate")
    ui.queue().launch(theme=theme, css=CUSTOM_CSS, inbrowser=True)
