from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

import torch

from rldata.oxe.bucket import discover_dataset_versions, discover_datasets_from_bucket
from rldata.oxe.utils import (
    ModalitySpec,
    flatten_structure,
    latest_version,
    normalize_version_key,
    tf_to_torch,
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


def _copy_tree(src: str, dst: str) -> None:
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


# ---------------------------------------------------------------------------
# OXEDataset
# ---------------------------------------------------------------------------

class OXEDataset(torch.utils.data.IterableDataset):
    """IterableDataset wrapper around an OXE TFDS builder.

    The full TFDS dataset is mirrored to the local cache directory on first
    use, then served from disk on subsequent runs.  If `episodes` is provided,
    only those episode indices are yielded during iteration; the full dataset
    is still cached so individual episodes can be accessed cheaply.

    TF tensors are converted to torch tensors on the fly inside __iter__ —
    nothing is pre-converted to torch before iteration.

    Cache directory (in priority order):
        1. `root` argument
        2. RLDATA_CACHE environment variable
        3. ~/.cache/rldata  (default)
    """

    def __init__(
        self,
        dataset_name: str = "droid",
        split: str = "train",
        version: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        shuffle_files: bool = False,
        root: Optional[str] = None,
        **as_dataset_kwargs: Any,
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
        self.shuffle_files = shuffle_files
        self.as_dataset_kwargs = dict(as_dataset_kwargs)
        self.dataset_path = dataset2path(dataset_name, version=version)
        self.root = _get_cache_dir(root)

        local_dir = self._local_tfds_dir()
        info_file = local_dir / "dataset_info.json"
        if not info_file.exists() or info_file.stat().st_size == 0:
            _copy_tree(self.dataset_path, str(local_dir))

        try:
            self.builder = tfds.builder_from_directory(builder_dir=str(local_dir))
            self.info = getattr(self.builder, "info", None)
            self.modalities = self._infer_modalities()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from '{local_dir}'. {error}"
            ) from error

    # ------------------------------------------------------------------
    # Internal: modality inference
    # ------------------------------------------------------------------

    def _infer_modalities(self) -> Dict[str, Dict[str, Any]]:
        """Derive modality info from builder.meta, falling back to builder.info.features."""
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_tfds_dir(self) -> Path:
        d = self.root / "oxe" / self.dataset_name
        if self.version:
            d = d / self.version
        return d

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset = self.builder.as_dataset(
            split=self.split,
            shuffle_files=False,
            **self.as_dataset_kwargs,
        )
        for episode in dataset.skip(idx).take(1):
            return _tf_to_torch(episode)
        raise IndexError(f"Episode index {idx} is out of range")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        dataset = self.builder.as_dataset(
            split=self.split,
            shuffle_files=self.shuffle_files,
            **self.as_dataset_kwargs,
        )

        if self.episodes is not None:
            selected = set(self.episodes)
            max_idx = max(selected)
            for stream_idx, episode in enumerate(dataset):
                if stream_idx in selected:
                    yield _tf_to_torch(episode)
                if stream_idx >= max_idx:
                    break
        else:
            for episode in dataset:
                yield _tf_to_torch(episode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
