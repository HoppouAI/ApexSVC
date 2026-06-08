"""Identity + structural pipeline tests. Skipped if checkpoints missing.
Run with `uv run pytest`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

CKPT_DIR = Path("checkpoints")


@pytest.mark.skipif(not (CKPT_DIR / "G_150k.pt").exists(),
                    reason="G_150k.pt not downloaded yet")
def test_synth_loads():
    import json
    from svc.synth import GeneratorNSF
    from svc.utils.tools import AttrDict

    cfg_path = Path("svc/synth/configs/g_150k.json")
    with open(cfg_path) as f:
        cfg = AttrDict(json.load(f))
    gen = GeneratorNSF(cfg)
    state = torch.load(CKPT_DIR / "G_150k.pt", map_location="cpu", weights_only=False)
    gen.load_state_dict(state["generator"])


def test_coarse_f0_roundtrip():
    from svc.pitch import coarse_f0
    f0 = np.array([0.0, 100.0, 220.0, 440.0, 880.0])
    bins = coarse_f0(f0)
    assert bins.dtype == np.int64
    assert bins.min() >= 1
    assert bins.max() <= 255
    # bin should monotonically increase with pitch (after the unvoiced 0)
    assert (np.diff(bins[1:]) > 0).all()


def test_knn_shape():
    from svc.match import knn_match
    pool = torch.randn(500, 1024)
    q = torch.randn(40, 1024)
    out = knn_match(q, pool, topk=4)
    assert out.shape == (40, 1024)
