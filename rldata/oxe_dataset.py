from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, RandomSampler, SliceSampler, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from rldata.oxe.bucket import discover_dataset_versions, discover_datasets_from_bucket
from rldata.oxe.memmap_builder import (
    build_combined_storage,
    build_missing_episodes,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from rldata.oxe.utils import (
    ModalitySpec,
    flatten_structure,
    latest_version,
    normalize_version_key,
    tf_to_torch,
    dict_to_tensordict,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

try:
    import tensorflow_datasets as tfds
except ImportError:  # pragma: no cover - optional dependency
    tfds = None


OXE_BUCKET_URL = "gs://gresearch/robotics"

_DATASET_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_TF_TENSOR_TYPES = (tf.Tensor,) if tf is not None else tuple()


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def _get_cache_dir(override: Optional[str] = None) -> Path:
    """Return the root cache directory.

    Priority: override argument → RLDATA_CACHE env var → ~/.cache/rldata
    """
    if override is not None:
        return Path(override)
    env = os.environ.get("RLDATA_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "rldata"


# ---------------------------------------------------------------------------
# Progress helpers (LeRobotDataset-style tqdm)
# ---------------------------------------------------------------------------

def _progress(
    iterable: Iterable,
    *,
    desc: str = "",
    unit: str = "it",
    total: Optional[int] = None,
    leave: bool = True,
) -> Iterable:
    if _tqdm_cls is not None:
        return _tqdm_cls(iterable, desc=desc, unit=unit, total=total, leave=leave, dynamic_ncols=True)
    return iterable


# ---------------------------------------------------------------------------
# TF stack guard
# ---------------------------------------------------------------------------

def _require_tf_stack() -> None:
    if tf is None or tfds is None:
        raise RuntimeError(
            "OXEDataset requires tensorflow and tensorflow-datasets. "
            "Install those packages to load OXE datasets."
        )


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

def _get_dataset_map(refresh: bool = False, dataset_name: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    global _DATASET_CACHE

    # Fast path: if a specific dataset is requested and already cached, return immediately.
    if not refresh and _DATASET_CACHE is not None and (
        dataset_name is None or dataset_name in _DATASET_CACHE
    ):
        return _DATASET_CACHE

    if _DATASET_CACHE is None:
        _DATASET_CACHE = {}

    # When a dataset name is known, try its direct GCS path first — avoids scanning
    # the entire bucket (hundreds of API calls) just to validate one dataset.
    if dataset_name is not None and (refresh or dataset_name not in _DATASET_CACHE):
        direct = discover_dataset_versions(tf, OXE_BUCKET_URL, dataset_name)
        if direct:
            _DATASET_CACHE[dataset_name] = direct
            return _DATASET_CACHE

    # Full bucket scan — used by list_datasets() or when direct lookup found nothing.
    if refresh or not _DATASET_CACHE:
        _DATASET_CACHE = discover_datasets_from_bucket(tf, OXE_BUCKET_URL)

    return _DATASET_CACHE


def list_datasets(refresh: bool = False) -> Dict[str, List[str]]:
    """Return {dataset_name: [versions]} sorted newest-first from the GCS bucket.

    Example: {'viola': ['0.1.0'], 'bridge': ['1.0.0'], ...}
    """
    return {
        name: sorted(versions.keys(), key=normalize_version_key, reverse=True)
        for name, versions in _get_dataset_map(refresh).items()
    }


def validate_dataset_name(dataset_name: str, version: Optional[str] = None) -> bool:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        return False
    if version is not None:
        return version in datasets[dataset_name]
    return True


def dataset2path(dataset_name: str, version: Optional[str] = None) -> str:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Use list_datasets() to see available datasets."
        )
    versions = datasets[dataset_name]
    if version is not None:
        if version not in versions:
            available = ", ".join(sorted(versions.keys(), key=normalize_version_key, reverse=True))
            raise ValueError(
                f"Version '{version}' not available for '{dataset_name}'. Available: {available}"
            )
        return versions[version]
    return versions[latest_version(list(versions.keys()))]


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _tf_to_torch(value: Any) -> Any:
    return tf_to_torch(value, _TF_TENSOR_TYPES)


def _flatten_structure(tree: Any, prefix: str = "") -> Dict[str, ModalitySpec]:
    return flatten_structure(tree, _TF_TENSOR_TYPES, prefix)


def _to_tensordict(episode: Any) -> Any:
    """Convert a raw TF episode to a TensorDict."""
    return dict_to_tensordict(_tf_to_torch(episode))


# ---------------------------------------------------------------------------
# GCS → local copy helpers
# ---------------------------------------------------------------------------

def _gcs_walk_files(src: str, dst: str) -> List[tuple]:
    pairs: List[tuple] = []
    src = src.rstrip("/")
    dst = dst.rstrip("/")
    for entry in tf.io.gfile.listdir(src):
        name = entry.strip("/")
        s = f"{src}/{name}"
        d = f"{dst}/{name}"
        if tf.io.gfile.isdir(s):
            pairs.extend(_gcs_walk_files(s, d))
        else:
            pairs.append((s, d))
    return pairs


def _needs_download(dst_file: str) -> bool:
    """Return True if dst_file is missing or empty (i.e. a failed partial download)."""
    if not tf.io.gfile.exists(dst_file):
        return True
    try:
        return tf.io.gfile.stat(dst_file).length == 0
    except Exception:
        return True


def _is_data_shard(filename: str) -> bool:
    """Return True if filename is a TFDS data shard (not a metadata file)."""
    return any(ext in filename for ext in (".tfrecord", ".riegeli", ".array_record"))


def _copy_tree(src: str, dst: str) -> None:
    """Download all files (metadata + data shards) from GCS to local dir."""
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    for src_file, dst_file in _progress(
        pairs, desc="Downloading dataset files", unit="file", total=len(pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


def _copy_metadata_only(src: str, dst: str) -> None:
    """Download only TFDS metadata files (dataset_info.json etc.), skipping data shards.

    Used when specific episodes are requested — the heavy shard files stay on
    GCS and are streamed on-demand during TED conversion.
    """
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    metadata_pairs = [(s, d) for s, d in pairs if not _is_data_shard(s)]
    for src_file, dst_file in _progress(
        metadata_pairs, desc="Downloading dataset metadata", unit="file", total=len(metadata_pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


# ---------------------------------------------------------------------------
# OXEDataset
# ---------------------------------------------------------------------------

class OXEDataset(BaseDatasetExperienceReplay):
    """OXE offline dataset as a TorchRL ``BaseDatasetExperienceReplay``.

    On first use the TFDS dataset is downloaded from the OXE GCS bucket and
    converted to TED (Trajectory Episode Data) format step-by-step, then
    persisted as memory-mapped tensors so subsequent runs skip both steps.
    Inheriting from ``BaseDatasetExperienceReplay`` (rather than bare
    ``ReplayBuffer``) provides:

    * **Immutability** — ``ImmutableDatasetWriter`` prevents accidental writes.
    * **``preprocess()``** — parallelised transform pipeline to normalise
      observations or fuse modalities, saving results to a new memmap.
    * **``delete()``** — clears the cached memmap from disk.
    * **``data_path`` / ``data_path_root``** — standardised path interface.
    * **Ecosystem fit** — recognised by TorchRL tooling the same way D4RL /
      Minari datasets are.

    TED layout per step::

        TensorDict({
            "observation":  TensorDict({...}),   # step[t] modalities (nested)
            "action":       Tensor,
            "done":         Tensor([1], bool),
            "terminated":   Tensor([1], bool),
            "next": TensorDict({
                "observation": TensorDict({...}), # step[t+1] obs (copy for last)
                "reward":      Tensor([1]),
                "done":        Tensor([1], bool),
                "terminated":  Tensor([1], bool),
            }),
            "collector": TensorDict({
                "traj_ids": Tensor(int64),        # episode index for SliceSampler
            }),
        })

    Usage::

        ds = OXEDataset("droid", episodes=[0, 1, 2], batch_size=32)
        batch = ds.sample()          # TensorDict(batch_size=[32])
        print(ds.num_episodes)       # 3
        print(len(ds))               # total steps

    Cache directory (priority order):
        1. ``root`` argument
        2. ``RLDATA_CACHE`` environment variable
        3. ``~/.cache/rldata``  (default)

    Args:
        dataset_name: OXE dataset name (e.g. ``"droid"``).
        split: TFDS split, e.g. ``"train"``.
        version: Specific dataset version; auto-selects latest when omitted.
        episodes: List of episode indices to include.  Only those episodes are
            converted; the full dataset is otherwise used.
        batch_size: Number of transitions returned by ``sample()``.
        slice_len: If set, ``sample()`` returns contiguous sub-trajectories of
            this length via ``SliceSampler``.
        root: Override cache root directory.
    """

    def __init__(
        self,
        dataset_name: str = "droid",
        split: str = "train",
        version: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        slice_len: Optional[int] = None,
        root: Optional[str] = None,
    ) -> None:
        dataset_name = dataset_name.strip("/")

        if not validate_dataset_name(dataset_name, version):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'"
                + (f" version '{version}'" if version else "")
                + f". Available datasets: {', '.join(sorted(list_datasets().keys()))}"
            )

        _require_tf_stack()

        self.dataset_name = dataset_name
        self.split = split
        self.version = version
        self.episodes: Optional[List[int]] = list(episodes) if episodes is not None else None
        self.dataset_path = dataset2path(dataset_name, version=version)
        self.root = _get_cache_dir(root)

        # ------------------------------------------------------------------
        # 1. Sync metadata JSON files from GCS if not already present
        # ------------------------------------------------------------------
        local_dir = self._local_tfds_dir()
        info_file = local_dir / "dataset_info.json"
        if not info_file.exists() or info_file.stat().st_size == 0:
            _copy_metadata_only(self.dataset_path, str(local_dir))

        try:
            self.builder = tfds.builder_from_directory(builder_dir=str(local_dir))
            self.info = getattr(self.builder, "info", None)
            self.modalities = self._infer_modalities()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from '{local_dir}'. {error}"
            ) from error

        # ------------------------------------------------------------------
        # 2. Determine which episodes to load; find what's missing from cache
        # ------------------------------------------------------------------
        episodes_dir = self._episodes_dir()

        if self.episodes is not None:
            selected = sorted(self.episodes)
        else:
            selected = list(range(self._get_total_episodes()))

        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]

        # ------------------------------------------------------------------
        # 3. Download and convert only missing episodes
        # ------------------------------------------------------------------
        if missing:
            if self.episodes is not None:
                # Episode-selective: stream missing episodes directly from GCS.
                # No shard files are written locally.
                builder_for_data = tfds.builder_from_directory(builder_dir=self.dataset_path)
            else:
                # Full dataset: download shards to local cache (idempotent),
                # then read locally for conversion.
                _copy_tree(self.dataset_path, str(local_dir))
                builder_for_data = self.builder

            build_missing_episodes(
                builder=builder_for_data,
                split=split,
                episodes_dir=episodes_dir,
                missing=missing,
                tf_tensor_types=_TF_TENSOR_TYPES,
            )

        # ------------------------------------------------------------------
        # 4. Build (or reuse) the combined memmap, then load it lazily.
        #    Peak RAM during build = one episode; during training = batch_size.
        # ------------------------------------------------------------------
        self._loaded_indices: List[int] = selected
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        if slice_len is not None:
            num_slices = max(1, batch_size // slice_len)
            sampler = SliceSampler(
                slice_len=slice_len,
                num_slices=num_slices,
                traj_key=("collector", "traj_ids"),
                end_key=("next", "done"),
            )
        else:
            sampler = RandomSampler()

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=ImmutableDatasetWriter(),
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # BaseDatasetExperienceReplay abstract interface
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        """Per-episode TED memmap directory for the current split."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root path for all cached data for this dataset."""
        return self._local_tfds_dir()

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_tfds_dir(self) -> Path:
        d = self.root / "oxe" / self.dataset_name
        if self.version:
            d = d / self.version
        return d

    def _episodes_dir(self) -> Path:
        return self._local_tfds_dir() / "episodes" / self.split

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        return self._local_tfds_dir() / "combined" / self.split / key

    def _get_total_episodes(self) -> int:
        """Return total episode count for the current split from builder.info."""
        if self.info is not None:
            splits = getattr(self.info, "splits", {})
            split_info = splits.get(self.split)
            if split_info is not None:
                count = getattr(split_info, "num_examples", None)
                if count is not None:
                    return int(count)
        raise RuntimeError(
            f"Cannot determine episode count for '{self.dataset_name}/{self.split}' "
            "from builder.info. Pass an explicit episodes list."
        )

    def _infer_modalities(self) -> Dict[str, Dict[str, Any]]:
        meta = getattr(self.builder, "meta", None)
        metadata_tree: Any = None

        if meta is not None:
            if isinstance(meta, Mapping):
                metadata_tree = meta
            elif hasattr(meta, "as_dict"):
                try:
                    metadata_tree = meta.as_dict()
                except Exception:
                    pass
            elif hasattr(meta, "features"):
                metadata_tree = meta.features

        if metadata_tree is None and self.info is not None:
            metadata_tree = getattr(self.info, "features", None)

        if metadata_tree is None:
            return {}

        flattened = _flatten_structure(metadata_tree)
        return {path: asdict(spec) for path, spec in flattened.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of episodes loaded into this dataset."""
        return len(self._loaded_indices)

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.modalities)

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.info is None:
            return {}
        return {
            "description": getattr(self.info, "description", ""),
            "features": getattr(self.info, "features", {}),
            "splits": getattr(self.info, "splits", {}),
        }
