# SVC-Experiments

Zero-shot singing voice conversion. Drop in a source audio and up to ~5 minutes of
reference audio of the target voice, get the source sung in the reference voice.
No per-voice training, no fine-tuning, no LoRAs. Pretrained models only.

## Approach

Retrieval-based, in the spirit of RVC's "index" feature but without the per-voice
training step. The pipeline is:

1. Extract content features from source and reference with **WavLM-Large**.
2. Extract per-frame **F0** (RMVPE) and loudness from the source.
3. For each source frame, find the **k nearest** reference frames in WavLM space
   and average them. The matched features carry the reference timbre.
4. Synthesise audio with the **NeuCoSVC2** FastSVC vocoder plus a neural
   harmonic filter driven by the source F0.

Because the converted features come directly from the reference audio rather than
a learned prior, the result avoids the over-smoothed quality typical of diffusion
based zero-shot SVC. A longer reference genuinely helps, up to roughly five
minutes, after which the matching pool saturates.

## Hardware

Tested on Windows with NVIDIA GPUs (CUDA 12.1). 6 GB VRAM is enough for inference.

## Setup

```powershell
# one shot setup, creates .venv, installs deps, downloads model weights
./setup.ps1
```

Or manually:

```powershell
uv venv
uv sync
uv run python scripts/download_models.py
```

## Usage

CLI:

```powershell
uv run svc convert --src path/to/source.wav --ref path/to/reference.wav --out output.wav
```

Multiple reference files (will be concatenated into one matching pool):

```powershell
uv run svc convert --src source.wav --ref ref1.wav --ref ref2.wav --ref ref3.wav --out output.wav
```

Gradio UI:

```powershell
uv run python app.py
```

## License

MIT (this repository). Bundled / vendored model code retains its original license,
see `THIRD_PARTY_NOTICES.md`.
