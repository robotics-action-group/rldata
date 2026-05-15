from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from rldata.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

_SENTINEL = "_complete.json"


def _progress(iterable, *, desc: str = "", unit: str = "it", total: Optional[int] = None):
    if _tqdm_cls is not None:
        return _tqdm_cls(iterable, desc=desc, unit=unit, total=total, leave=True, dynamic_ncols=True)
    return iterable


def is_memmap_complete(memmap_dir: Path) -> bool:
    return (memmap_dir / _SENTINEL).exists()


def build_memmap(
    builder: Any,
    split: str,
    memmap_dir: Path,
    episodes: Optional[List[int]],
    tf_tensor_types: Tuple[type, ...],
) -> int:
    """Convert selected episodes from a TFDS builder to TED-format memmap storage.

    Steps from every selected episode are converted to TED TensorDicts and
    stacked into a single TensorDict that is memory-mapped to ``memmap_dir``.
    A sentinel file ``_complete.json`` is written on success.

    Returns the total number of steps written.
    """
    dataset = builder.as_dataset(split=split, shuffle_files=False)

    if episodes is not None:
        selected = set(episodes)
        max_idx = max(selected)
        episode_iter = (
            (global_idx, ep)
            for global_idx, ep in enumerate(dataset)
            if global_idx <= max_idx
        )
    else:
        episode_iter = enumerate(dataset)

    all_steps: list = []
    n_episodes = 0

    for global_idx, episode in _progress(episode_iter, desc="Converting episodes to TED", unit="ep"):
        if episodes is not None and global_idx not in selected:
            continue
        steps = episode_to_ted_steps(episode, n_episodes, tf_tensor_types)
        all_steps.extend(steps)
        n_episodes += 1

    if not all_steps:
        raise ValueError("No steps found; check dataset name, split, and episodes filter.")

    tensors_dir = memmap_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    all_td = torch.stack(all_steps)  # TensorDict(batch_size=[N_steps])
    all_td.memmap_(str(tensors_dir))

    n_steps = len(all_td)
    (memmap_dir / _SENTINEL).write_text(
        json.dumps({"n_steps": n_steps, "n_episodes": n_episodes})
    )
    return n_steps
