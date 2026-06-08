# Third-party notices

ApexSVC's own code is licensed under AGPL-3.0-or-later (see `LICENSE`). The
components listed below are vendored, wrapped, or pulled at runtime. Each keeps
its original license and authorship; the SPDX identifier next to each entry is
the upstream license, not ApexSVC's. Visit the linked upstreams for full
license texts.

## NeuCoSVC / NeuCoSVC2
- Upstream: https://github.com/thuhcsi/NeuCoSVC (branch `NeuCoSVC2`)
- Paper: Sha et al., "Neural Concatenative Singing Voice Conversion", arXiv:2312.04919
- License: MIT
- Files vendored: the NSF HiFi-GAN generator (`GeneratorNSF`, source-filter
  excited HiFi-GAN), kNN matcher, and small audio utility helpers, under
  `svc/synth/`, `svc/match/`, and `svc/utils/`. The training code, dataset
  scaffolding, REAPER pitch wrapper, and Phoneme Hallucinator are NOT vendored.

## WavLM (Microsoft)
- Upstream: https://github.com/microsoft/unilm/tree/master/wavlm
- License: MIT
- Used as the content encoder (layer 6). Weights pulled at first run from the
  `microsoft/wavlm-large` Hugging Face repository.

## TorchFCPE
- Upstream: https://github.com/CNChTu/FCPE (PyPI: `torchfcpe`)
- License: MIT
- Default F0 extractor. Installed as a regular pip dependency, weights are
  bundled in the wheel.

## praat-parselmouth (optional F0 backend)
- Upstream: https://github.com/YannickJadoul/Parselmouth
- License: GPL-3.0-or-later (Praat itself), Parselmouth bindings under MIT.
- Wraps Praat's autocorrelation pitch tracker. Used when `--f0-method praat` is
  selected.

## librosa (optional F0 backend)
- Upstream: https://github.com/librosa/librosa
- License: ISC
- Provides PYIN, used when `--f0-method pyin` is selected.

## kNN-VC
- Upstream: https://github.com/bshall/knn-vc
- License: MIT
- Method inspiration for the retrieval-style matcher (Baas et al., "Voice
  Conversion With Just Nearest Neighbors", InterSpeech 2023). No code vendored,
  just the technique.

## Silero VAD
- Upstream: https://github.com/snakers4/silero-vad
- License: MIT
- Used for reference-audio voice activity detection. Pulled as a pip dependency,
  weights downloaded on first run.

## PyTorch / torchaudio
- Upstream: https://pytorch.org
- License: BSD-3-Clause
- Pinned to the cu121 wheel index in `pyproject.toml` for Windows CUDA support.

## Gradio
- Upstream: https://github.com/gradio-app/gradio
- License: Apache-2.0
- Powers the local web UI in `app.py`.

## Other runtime dependencies
Standard scientific Python stack (NumPy, SciPy, soundfile, resampy, tqdm,
click, Hugging Face huggingface_hub / transformers / safetensors / omegaconf,
gdown). All used as regular pip packages under their respective permissive
licenses (BSD, MIT, Apache-2.0). See `pyproject.toml` for the full pinned list.

