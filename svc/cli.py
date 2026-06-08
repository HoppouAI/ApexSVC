"""CLI: `uv run svc convert --src ... --ref ... --out ...`"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .pipeline import DEFAULT_CHECKPOINT_DIR, ConversionConfig, SVCPipeline


@click.group()
@click.option("--verbose", "-v", count=True, help="-v info, -vv debug")
def main(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--src", "source", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Source singing audio.")
@click.option("--ref", "references", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, multiple=True,
              help="Reference audio file(s). Pass multiple --ref to merge into one matching pool.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Output wav path.")
@click.option("--topk", type=int, default=4, show_default=True,
              help="Number of nearest neighbours to average per source frame.")
@click.option("--pitch-shift", "pitch_shift_semitones", type=float, default=None,
              help="Manual semitone shift. If omitted, source mean F0 is auto-aligned to reference.")
@click.option("--speech-enroll", is_flag=True,
              help="Treat reference as speech (not singing): multiplies auto pitch factor by 1.2x.")
@click.option("--alpha", type=float, default=0.0, show_default=True,
              help="Blend factor from retrieved features (0) back to raw source features (1).")
@click.option("--no-vad-trim", "vad_trim_reference", flag_value=False, default=True,
              help="Disable silero-vad trimming of reference audio.")
@click.option("--f0-method", "f0_method",
              type=click.Choice(["fcpe", "praat", "pyin", "median"]), default="fcpe",
              show_default=True,
              help="F0 tracker: fcpe (neural, singing-grade, default), praat (fast classical), pyin (slow robust), median (both classical).")
@click.option("--checkpoint-dir", type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_CHECKPOINT_DIR, show_default=True)
@click.option("--device", type=str, default=None,
              help="Override device (cuda / cpu). Auto-detects by default.")
def convert(source: Path, references: tuple[Path, ...], out_path: Path, topk: int,
            pitch_shift_semitones: float | None, speech_enroll: bool, alpha: float,
            vad_trim_reference: bool, f0_method: str, checkpoint_dir: Path,
            device: str | None) -> None:
    """Convert SRC to sound like REF, write to OUT."""
    pipe = SVCPipeline(checkpoint_dir=checkpoint_dir, device=device)
    cfg = ConversionConfig(
        topk=topk,
        pitch_shift_semitones=pitch_shift_semitones,
        speech_enroll=speech_enroll,
        alpha=alpha,
        vad_trim_reference=vad_trim_reference,
        f0_method=f0_method,
        device=str(pipe.device),
        checkpoint_dir=checkpoint_dir,
    )

    from tqdm.auto import tqdm

    bar = tqdm(total=100, bar_format="{l_bar}{bar}| {n}/100 [{elapsed}<{remaining}] {postfix}",
               ncols=90, leave=True)
    last_pct = {"v": 0}

    def _prog(frac: float, desc: str) -> None:
        pct = int(max(0.0, min(1.0, frac)) * 100)
        delta = pct - last_pct["v"]
        if delta > 0:
            bar.update(delta)
            last_pct["v"] = pct
        bar.set_postfix_str(desc[:60])

    try:
        wav, sr = pipe.convert(source, list(references), cfg=cfg, progress=_prog)
    finally:
        bar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    sf.write(str(out_path), wav, sr)
    click.echo(f"Wrote {out_path} ({wav.shape[0] / sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
