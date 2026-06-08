<div align="center">

# ApexSVC

**Zero shot singing voice conversion. Drop in a song, drop in a reference voice, get the song sung in that voice. No training, no fine tuning, no LoRAs.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://docs.astral.sh/uv/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00.svg)](https://www.gradio.app/)
[![Platform](https://img.shields.io/badge/platform-Windows-0078d6.svg?logo=windows&logoColor=white)](https://www.microsoft.com/windows)

</div>

> ApexSVC is a retrieval based zero shot SVC system. It uses WavLM for content features, FCPE for pitch, k nearest neighbour matching against the reference voice, and an NSF HiFi GAN style vocoder for synthesis. The whole thing runs at multiple times realtime on a 12 GB consumer GPU.

---

## Why ApexSVC

Most SVC tools either need you to train a model per voice (RVC, so vits svc) or rely on a diffusion prior that smears timbre. ApexSVC is closer in spirit to k nearest neighbour voice conversion: the converted features come straight from the reference audio at inference time, so it sounds like the reference instead of an averaged guess at it.

| | ApexSVC | RVCv2 (mainline) |
| :--- | :---: | :---: |
| Per voice training required | No | Yes |
| Setup time per voice | Seconds (encode reference) | 15 min to 2 hours (GPU and dataset dependent) |
| Reference / dataset needed | 30 sec to 5 min reference | 10 min minimum, more is better |
| Content encoder | WavLM Large (layer 6) | ContentVec (HuBERT based) |
| Pitch tracker | FCPE (neural) | RMVPE (default), pm / harvest / crepe optional |
| Vocoder | NSF HiFi GAN (NeuCoSVC2) | VITS with HiFi GAN decoder |
| Realtime inference | Yes on consumer GPU | Yes (90 to 170 ms with ASIO) |
| Out of distribution phonemes | Bounded by reference | Generalises better (trained model) |

---

## Samples

Hear it for yourself. The actual reference fed to ApexSVC was a ~7 minute dataset file (around 5 minutes of usable voice after VAD trims silence and breaths). The 32 second clip below is just a quick voice preview so you know what Gabriel actually sounds like before listening to the conversions. Zero shot for ApexSVC, fully trained model for RVC.

**Voice preview (Gabriel, 32s)** &mdash; just so you have something to compare against. NOT the reference fed to the model.

<video src="https://github.com/HoppouAI/ApexSVC/raw/main/samples/Gabriel-Voice-Reference.mp4" controls width="100%"></video>

**Source vocal** &mdash; the original isolated stem before any conversion.

<video src="https://github.com/HoppouAI/ApexSVC/raw/main/samples/ApexSVC-Gabriel-Vocals-Source.mp4" controls width="100%"></video>

**RVCv2 (trained model)** &mdash; for comparison, RVCv2 conversion using a fully trained model on the same voice.

<video src="https://github.com/HoppouAI/ApexSVC/raw/main/samples/ApexSVC-Gabriel-RVC-Vocals.mp4" controls width="100%"></video>

**ApexSVC zero-shot** &mdash; same source, no training, reference was a ~7 minute dataset clip (about 5 min usable after VAD).

<video src="https://github.com/HoppouAI/ApexSVC/raw/main/samples/ApexSVC-Gabriel-1-Vocals-Converted.mp4" controls width="100%"></video>

**ApexSVC + instrumental** &mdash; the ApexSVC vocal mixed back over the original instrumental.

<video src="https://github.com/HoppouAI/ApexSVC/raw/main/samples/ApexSVC-Gabriel-1-Full.mp4" controls width="100%"></video>

---

## Features

| Feature | What it does |
| :--- | :--- |
| **Zero shot** | No fine tuning, no per voice training. One reference clip is enough. |
| **WavLM Large content** | Layer 6 features, robust to mic, room, language. |
| **FCPE pitch** | Neural F0 tracker, faster and steadier than RMVPE on noisy vocals. |
| **kNN retrieval** | Cosine similarity match against the reference pool, top k averaged. |
| **NSF HiFi GAN vocoder** | 24 kHz output, source filter excited, no over smoothing. |
| **Auto pitch shift** | Detects a sensible pitch shift from reference median F0, snaps to semitones, octave folds. |
| **Pro autotune** | Strength slider plus retune time constant. Glides between target notes instead of staircasing. |
| **Voiceless protect** | Optional source bleed on consonants for cleaner diction. |
| **RMS envelope match** | Optionally copies the source loudness contour onto the output. |
| **Loudness normalize** | LUFS target so output sits at a mix friendly level. |
| **Voice profiles** | Encode a reference once, save to disk, reload in milliseconds. |
| **Reference cache** | Memory and disk cache so re running the same reference is instant. |
| **VRAM controls** | Free reserved torch VRAM and drop the reference cache from the UI. |
| **Gradio UI + CLI** | Same pipeline behind both, no drift. |

---

## Hardware

| | Tested | Likely works | Won't work |
| :--- | :---: | :---: | :---: |
| **GPU** | RTX 3060 12 GB | Anything with 6 GB+ CUDA | <4 GB VRAM |
| **OS** | Windows 11 | Linux, macOS (CPU only) | n/a |
| **CUDA** | 12.1 | 11.8 with custom torch | ROCm not tested |
| **RAM** | 32 GB | 16 GB | 8 GB tight |

> Inference only needs roughly 6 GB of VRAM. WavLM Large is the heaviest piece.

---

## Compatibility

| Platform | Status | Notes |
| :--- | :---: | :--- |
| Windows 10 / 11 | Tested, fully supported | Primary dev target. CUDA 12.1 wheels of torch are pinned in `pyproject.toml`. |
| Linux | Should work, not tested | A `setup.sh` is included for parity. File issues if anything breaks. |
| macOS | Untested | Pure CPU only, expect slow inference. No CUDA wheels exist for Apple Silicon. |

> Heads up: ApexSVC has only been tested on Windows. The Linux path is provided for convenience and parity, but no QA passes have run there. Bug reports welcome.

---

## Setup

### Windows (recommended path)

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), git, and an NVIDIA GPU with up to date drivers.

```powershell
# install uv if you don't already have it
winget install --id=astral-sh.uv -e

# clone and enter
git clone https://github.com/HoppouAI/ApexSVC.git
cd ApexSVC

# one shot setup, creates .venv, installs deps, downloads model weights
./setup.ps1
```

Or do it by hand:

```powershell
uv venv
uv sync
uv run python scripts/download_models.py
```

### Linux (untested)

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/HoppouAI/ApexSVC.git
cd ApexSVC

chmod +x setup.sh
./setup.sh
```

If `setup.sh` falls over, the manual path is the same as Windows:

```bash
uv venv
uv sync
uv run python scripts/download_models.py
```

### What gets installed

| Component | Source | Size |
| :--- | :--- | ---: |
| WavLM Large | huggingface.co/microsoft/wavlm-large | ~1.2 GB |
| NSF HiFi GAN G_150k checkpoint | NeuCoSVC2 release | ~150 MB |
| FCPE pitch tracker | torchfcpe pip package | ~30 MB |
| Silero VAD | pip package, downloads on first run | small |

---

## Usage

### CLI

Bare minimum:

```powershell
uv run svc convert --src path/to/source.wav --ref path/to/reference.wav --out output.wav
```

Multiple references get concatenated into one matching pool:

```powershell
uv run svc convert --src source.wav --ref ref1.wav --ref ref2.wav --ref ref3.wav --out out.wav
```

With pro autotune:

```powershell
uv run svc convert --src source.wav --ref voice.wav --out out.wav `
  --autotune-strength 0.7 --autotune-retune-ms 60
```

Full option list with `uv run svc convert --help`.

### Gradio UI

```powershell
uv run python app.py
```

Then open the URL it prints. The UI has tabs for **Convert**, **Guide**, **Tips**, and **Troubleshooting**. Drop a source audio in the source slot, drop a reference (or several) in the reference slot, hit **Convert**.

### Voice profiles

Encoding a reference takes a few seconds. If you reuse the same voice often, save it as a profile and reload in milliseconds:

```powershell
# save
uv run svc save-voice --name gabriel --ref inputs/Gabrielv3-dataset.wav

# convert using a saved profile (when supported in your CLI version)
# you can also just rely on the on disk reference cache, which keys off file hash
```

---

## Settings cheat sheet

| Setting | Default | When to change |
| :--- | :---: | :--- |
| `--f0-method` | `fcpe` | Use `pyin` if FCPE produces wobbles on very breathy vocals. |
| `--topk` | `4` | Higher = smoother and safer, lower = closer to source nuance. 2 to 8 is the useful range. |
| `--alpha` | `0.0` | Slight blend of source features. 0 keeps full target voice, 0.1 to 0.2 helps if pronunciation goes off. |
| `--protect` | `0.0` | Bleed source onto voiceless frames. Try 0.3 to 0.5 if consonants sound mushy. |
| `--rms-mix-rate` | `0.0` | Copy source loudness envelope. Good for songs with big dynamic swings. |
| `--f0-filter-radius` | `1` | Median smooth pitch outliers. Bump to 3 if pitch glitches. |
| `--autotune-strength` | `0.0` | 0 off, 0.3 subtle pitch fix, 0.7 noticeable, 1.0 full snap. |
| `--autotune-retune-ms` | `60` | Smaller is snappier (T Pain style), larger is smoother and more natural. |
| `--pitch-shift` | auto | Override the auto detected shift in semitones. |

---

## FAQ

<details>
<summary><b>How long should the reference voice be?</b></summary>

Anywhere from 30 seconds works. Quality keeps climbing until roughly 5 minutes, after which the matching pool saturates and you stop getting better timbre out of more data. Multiple short clips concatenated together work fine and are often easier to gather.
</details>

<details>
<summary><b>Why does the converted voice sound a little off pitch?</b></summary>

Auto pitch shift estimates the right octave from the reference's median F0 and snaps to a semitone. If the source vocal is unusual (whisper, growl, very low energy) the median can land wrong. Use `--pitch-shift N` to force a specific shift in semitones, or open Gradio and uncheck **auto pitch**.
</details>

<details>
<summary><b>How does this compare to RVC?</b></summary>

RVC trains a small VITS model per voice. That gives the highest fidelity ceiling and lets the model generalise to phonemes the speaker never produced in the training data, but takes 15 minutes to a couple of hours of GPU time per character (depending on your GPU, dataset size, and epoch count) and needs roughly 10 minutes or more of curated audio. ApexSVC is zero shot: it pulls features straight from the reference audio at inference time, so a one minute reference is enough to get going. The trade off is that ApexSVC is bounded by what's actually in the reference clip, while a trained RVC model can extrapolate.
</details>

<details>
<summary><b>Can I run this on CPU?</b></summary>

Technically yes, install the CPU torch wheels instead of the cu121 ones. Realistically, expect minutes per second of audio. Use a GPU.
</details>

<details>
<summary><b>Why is my first conversion slow?</b></summary>

Two reasons. First, WavLM Large is loaded on first call, that's a one time ~5 second hit. Second, the reference is encoded the first time you use it. Subsequent runs with the same reference hit the disk cache and skip the encode entirely. The Gradio UI keeps the pipeline loaded between conversions.
</details>

<details>
<summary><b>VRAM stays high after conversion. Is that a leak?</b></summary>

No. PyTorch's caching allocator keeps reserved memory around for speed. The Gradio UI has a **VRAM** accordion with buttons to release reserved memory and drop the reference cache. Task Manager shows reserved, not in use, so it's normal for it to look high.
</details>

<details>
<summary><b>What audio formats are supported?</b></summary>

Anything `soundfile` can read, which covers wav, flac, ogg, and most common formats. Output is always 24 kHz wav. Convert mp3 / m4a sources to wav first if you hit decoding errors.
</details>

<details>
<summary><b>Does this work for speech, not just singing?</b></summary>

Yes. Pass `--speech-enroll` to skip pitch shifting heuristics that assume the source is sung. Singing references are still fine, the flag only affects how the source is interpreted.
</details>

<details>
<summary><b>Can I convert a full song with instrumentals?</b></summary>

Separate the vocals first using something like UVR (Ultimate Vocal Remover). ApexSVC operates on isolated vocals. Mix the converted vocal back over the original instrumental in your DAW.
</details>

---

## Approach

Retrieval based, in the spirit of RVC's index feature but without the per voice training step. The pipeline:

1. Extract content features from source and reference with **WavLM Large** (layer 6).
2. Extract per frame **F0** with FCPE, plus loudness from the source.
3. For each source frame, find the **k nearest** reference frames in WavLM space and average them. The matched features carry the reference timbre.
4. Synthesise audio with the **NSF HiFi GAN** vocoder, source filter excited by a sine bank driven by the source F0.
5. Optional post processing: loudness normalisation, RMS envelope match, autotune.

Because the converted features come directly from the reference audio rather than a learned prior, the result avoids the over smoothed quality typical of diffusion based zero shot SVC. A longer reference genuinely helps, up to roughly five minutes, after which the matching pool saturates.

---

## Project layout

```
ApexSVC/
  app.py                 Gradio UI
  svc/
    cli.py               click based CLI entry point (`uv run svc ...`)
    pipeline.py          top level convert() + ConversionConfig
    content/             WavLM encoder
    pitch/               FCPE / praat / pyin extractors + autotune
    match/               kNN retrieval
    synth/               NSF HiFi GAN vocoder (vendored from NeuCoSVC2)
    utils/               audio io, voiced area extraction
  preprocess/            Silero VAD wrapper
  scripts/
    download_models.py   one shot model fetcher
    bench_*.py           profiling + RTF benchmarks
  third_party/           vendored model code (see THIRD_PARTY_NOTICES.md)
  tests/                 pytest suite
  inputs/   outputs/     audio dropbox folders
  voices/                saved voice profiles + on disk reference cache
  setup.ps1  setup.sh    one shot environment setup
  pyproject.toml         uv managed deps, pinned cu121 torch index
```

---

## Acknowledgements

| Component | Credit |
| :--- | :--- |
| WavLM | Microsoft Research |
| FCPE | torchfcpe authors |
| NSF HiFi GAN vocoder | NeuCoSVC2 (THUHCSI) |
| kNN VC method | Inspired by Baas et al, "Voice Conversion With Just Nearest Neighbors" |
| Silero VAD | Silero AI |
| RVC | Reference point for the SVC ecosystem |

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for license details on vendored code.

---

## Ethical use and disclaimer

ApexSVC is a voice conversion tool. It can clone the timbre of any voice it has a reference clip for, including real people. Please use it responsibly.

**Don't:**
- Clone someone's voice without their explicit, informed consent. That includes celebrities, public figures, friends, family, voice actors, content creators, deceased people, and anyone else.
- Use it for fraud, harassment, defamation, identity theft, non consensual sexual content, fake "evidence", scams, political disinformation, or any other harmful or illegal purpose.
- Use it to violate the terms of service of any platform you upload converted audio to.
- Misrepresent generated audio as authentic. If you publish a converted clip, label it clearly so listeners know it's synthesised.

**Do:**
- Get written permission before cloning a real person's voice, even your own friends.
- Respect copyright on source audio and reference recordings. Owning a song file does not grant you the right to redistribute a converted version.
- Disclose AI involvement in anything you publish, both for the audience's sake and to stay on the right side of platform rules.
- Check the laws in your jurisdiction. Several countries and US states now have specific statutes around deepfake audio, voice cloning, election content, and the right of publicity. Ignorance is not a defense.

**No watermarking.** ApexSVC does not embed any audible or inaudible watermark in its output. There is no automatic provenance signal, no "this was generated" tag, nothing. If you need watermarking for compliance reasons, add it yourself before distribution.

**No content moderation.** ApexSVC is a local tool. It does not phone home, does not screen reference voices, does not block any specific names or personas. Whatever you feed it gets converted.

**No liability.** ApexSVC is provided "as is" without warranty of any kind, express or implied. The authors and contributors are not responsible for how you use it, what you generate with it, or any harm, legal trouble, financial loss, or other damages arising from its use or misuse. You are fully responsible for your own outputs and for complying with all applicable laws and platform rules.

If your use case feels sketchy, it probably is. Don't be the reason this kind of tool gets banned for everyone else.

---

## License

ApexSVC's own code is licensed under the **GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later). See [`LICENSE`](LICENSE) for the full text.

In plain English: you can use, modify, and redistribute ApexSVC freely, but if you run a modified version as a network service (web app, hosted API, Discord bot, etc) you have to make your modified source available to users of that service. For local desktop / CLI use it behaves like a normal copyleft license.

Vendored / bundled model code keeps its original (mostly MIT and Apache style) licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the per-component breakdown.

