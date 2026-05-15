from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Set, Tuple

import torch

from rldata.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

_EPISODE_SENTINEL = "_steps.json"


def is_episode_cached(episode_dir: Path) -> bool:
    """Return True if this episode has already been converted and saved."""
    return (episode_dir / _EPISODE_SENTINEL).exists()


def _build_one_episode(
    episode: Any,
    global_idx: int,
    episode_dir: Path,
    tf_tensor_types: Tuple[type, ...],
) -> int:
    """Convert one episode to TED steps and memmap it to episode_dir.

    Uses global_idx as traj_ids so episode identity is consistent across
    different OXEDataset instances that may share the same cache.

    Returns the number of steps written.
    """
    steps = episode_to_ted_steps(episode, global_idx, tf_tensor_types)
    if not steps:
        return 0
    episode_dir.mkdir(parents=True, exist_ok=True)
    td = torch.stack(steps)
    td.memmap_(str(episode_dir))
    n_steps = len(td)
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
    return n_steps


def build_missing_episodes(
    builder: Any,
    split: str,
    episodes_dir: Path,
    missing: List[int],
    tf_tensor_types: Tuple[type, ...],
) -> None:
    """Convert and cache only the episodes in ``missing`` (global indices).

    Streams the TFDS dataset once up to ``max(missing)``, skipping episodes
    that are not requested.  Already-cached episodes in ``missing`` are also
    skipped (safe to call if a previous run was interrupted mid-way).

    Args:
        builder: TFDS builder (local or GCS) to read episode data from.
        split: Dataset split name (e.g. ``"train"``).
        episodes_dir: Root directory for per-episode memmaps
            (``~/.cache/rldata/oxe/{name}/episodes/{split}/``).
        missing: Global episode indices that are not yet cached.
        tf_tensor_types: TF tensor types tuple for conversion.
    """
    if not missing:
        return

    missing_set: Set[int] = set(missing)
    max_idx = max(missing_set)
    dataset = builder.as_dataset(split=split, shuffle_files=False)

    pbar = (
        _tqdm_cls(total=len(missing_set), desc="Converting episodes to TED", unit="ep", dynamic_ncols=True)
        if _tqdm_cls is not None
        else None
    )

    for global_idx, episode in enumerate(dataset):
        if global_idx > max_idx:
            break
        if global_idx not in missing_set:
            continue
        episode_dir = episodes_dir / str(global_idx)
        if not is_episode_cached(episode_dir):
            _build_one_episode(episode, global_idx, episode_dir, tf_tensor_types)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
