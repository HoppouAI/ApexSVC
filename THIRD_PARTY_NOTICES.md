# Third-party notices

This project vendors / wraps the following open-source components. Each retains
its original license and authorship; see linked upstreams for details.

## NeuCoSVC / NeuCoSVC2
- Upstream: https://github.com/thuhcsi/NeuCoSVC (branch `NeuCoSVC2`)
- Paper: Sha et al., "Neural Concatenative Singing Voice Conversion", arXiv:2312.04919
- Files vendored: the FastSVC synthesiser, neural harmonic generator, and kNN matcher
  modules (under `svc/synth/` and `svc/match/`). The training code, dataset
  scaffolding, REAPER pitch wrapper, and Phoneme Hallucinator are NOT vendored.

## WavLM (Microsoft)
- Upstream: https://github.com/microsoft/unilm/tree/master/wavlm
- Used as the content encoder. Weights pulled from the `microsoft/wavlm-large`
  Hugging Face repository at runtime.

## RMVPE
- Upstream: https://github.com/yxlllc/RMVPE (and other community ports)
- Used as the F0 extractor in place of REAPER. Weights pulled at first run.

## kNN-VC
- Upstream: https://github.com/bshall/knn-vc
- Method inspiration for the retrieval-style matcher.

## Silero VAD
- Upstream: https://github.com/snakers4/silero-vad
- Used for reference-audio voice activity detection.
