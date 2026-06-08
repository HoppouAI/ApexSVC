"""WavLM model (vendored verbatim from upstream NeuCoSVC2 / Microsoft WavLM).

The original WavLM-Large .pt checkpoint shipped by Microsoft includes a `cfg`
field that this WavLM class understands directly; the HuggingFace WavLMModel
uses a different config schema and weight mapping, so we keep the upstream
loader.
"""

from .WavLM import WavLM, WavLMConfig

__all__ = ["WavLM", "WavLMConfig"]
