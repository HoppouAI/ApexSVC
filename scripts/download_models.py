"""Download required model checkpoints into ./checkpoints/.

- WavLM-Large.pt   : from the s3prl HF mirror of the original Microsoft ckpt
                     (the original Microsoft Azure link 401s without auth).
- G_150k.pt        : NeuCoSVC2 generator, official Google Drive release.

If the files already exist they are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

CKPT_DIR = Path("checkpoints")
WAVLM_REPO = "s3prl/converted_ckpts"
WAVLM_FILE = "wavlm_large.pt"
G_150K_GDRIVE_ID = "1yDnT4Ah8Nlzq3QIff4ur4rz5CVpwYoip"


def fetch_wavlm() -> Path:
    target = CKPT_DIR / "WavLM-Large.pt"
    if target.exists():
        print(f"[skip] {target} already present")
        return target
    print(f"[wavlm] downloading from hf://{WAVLM_REPO}/{WAVLM_FILE} ...")
    cached = hf_hub_download(repo_id=WAVLM_REPO, filename=WAVLM_FILE,
                             cache_dir=str(CKPT_DIR / "_hf_cache"))
    # symlink/copy to canonical name
    import shutil
    shutil.copy(cached, target)
    print(f"[wavlm] -> {target}")
    return target


def fetch_g_150k() -> Path:
    target = CKPT_DIR / "G_150k.pt"
    if target.exists():
        print(f"[skip] {target} already present")
        return target
    print(f"[g150k] downloading from Google Drive id={G_150K_GDRIVE_ID} ...")
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed. run `uv sync` first.")
    url = f"https://drive.google.com/uc?id={G_150K_GDRIVE_ID}"
    gdown.download(url, str(target), quiet=False)
    print(f"[g150k] -> {target}")
    return target


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_wavlm()
    fetch_g_150k()
    print("\nAll checkpoints ready in", CKPT_DIR.resolve())


if __name__ == "__main__":
    main()
