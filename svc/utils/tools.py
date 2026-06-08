"""Small grab bag of helpers shared by the vendored synth + matcher code."""

import torch


class AttrDict(dict):
    """dict that exposes keys as attributes; used by the synth config loader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def fast_cosine_dist(source_feats: torch.Tensor, matching_pool: torch.Tensor,
                     device: str | torch.device = "cpu") -> torch.Tensor:
    """Pairwise cosine distance between (N1, D) and (N2, D) tensors. Same impl
    as upstream kNN-VC / NeuCoSVC for parity with their checkpoint expectations.
    """
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = (
        -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0] ** 2
        + source_norms[:, None] ** 2
        + matching_norms[None] ** 2
    )
    dotprod /= 2
    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
    return dists
