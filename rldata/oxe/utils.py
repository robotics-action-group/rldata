from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ModalitySpec:
    path: str
    kind: str
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    source: str


def normalize_version_key(version: str) -> Tuple[int, ...]:
    parts = [int(chunk) for chunk in re.split(r"[^0-9]+", version) if chunk]
    return tuple(parts) if parts else (0,)


def latest_version(versions: Sequence[str]) -> str:
    return sorted(versions, key=normalize_version_key)[-1]


def tf_to_torch(value: Any, tf_tensor_types: Tuple[type, ...] = ()) -> Any:
    """Convert a nested structure of TF tensors / numpy arrays to PyTorch tensors.

    Every numeric leaf becomes a torch.Tensor (zero-copy for numpy arrays via
    torch.from_numpy).  Non-numeric leaves (strings, bytes) are decoded to str
    since they cannot be represented as tensors.
    """
    if tf_tensor_types and isinstance(value, tf_tensor_types):
        value = value.numpy()  # fall through to numpy / bytes handling

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U", "O"}:  # string / object → decode to str
            decoded = value.flat[0] if value.ndim == 0 else value
            if value.dtype.kind == "S":
                return value.astype(str).tolist()
            return value.tolist()
        return torch.from_numpy(np.ascontiguousarray(value))

    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.bytes_):
            return value.item().decode("utf-8") if isinstance(value.item(), bytes) else str(value.item())
        return torch.as_tensor(value.item())

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return bytes(value)

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return torch.tensor(value)

    if isinstance(value, (int, float)):
        return torch.tensor(value)

    if isinstance(value, Mapping):
        return {k: tf_to_torch(v, tf_tensor_types) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tf_to_torch(v, tf_tensor_types) for v in value)
    if isinstance(value, list):
        return [tf_to_torch(v, tf_tensor_types) for v in value]

    return value


def infer_kind(path: str, value: Any = None) -> str:
    lowered = path.lower()
    if any(token in lowered for token in ("image", "camera", "rgb", "video", "frame")):
        return "image"
    if any(token in lowered for token in ("language", "instruction", "text", "caption", "prompt")):
        return "text"
    if any(token in lowered for token in ("action", "policy", "torque", "velocity")):
        return "action"
    if any(token in lowered for token in ("observation", "state", "proprio", "joint", "pose", "ee")):
        return "state"
    if isinstance(value, (str, bytes, bytearray)):
        return "text"
    if isinstance(value, np.ndarray) and value.ndim >= 3:
        return "image"
    return "generic"


def shape_and_dtype(value: Any, tf_tensor_types: Tuple[type, ...]) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    if isinstance(value, tf_tensor_types):
        shape = tuple(int(dim) if dim is not None else -1 for dim in value.shape)
        dtype = value.dtype.name
        return shape, dtype
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape), str(value.dtype)
    if isinstance(value, np.ndarray):
        return tuple(int(dim) for dim in value.shape), str(value.dtype)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = tuple(int(dim) for dim in value.shape)
        except Exception:
            shape = None
        return shape, str(value.dtype)
    return None, None


def flatten_structure(
    tree: Any,
    tf_tensor_types: Tuple[type, ...],
    prefix: str = "",
) -> Dict[str, ModalitySpec]:
    flattened: Dict[str, ModalitySpec] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_structure(value, tf_tensor_types, child_prefix))
        return flattened
    if isinstance(tree, (list, tuple)) and tree and not isinstance(tree[0], (bytes, bytearray, str)):
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            flattened.update(flatten_structure(value, tf_tensor_types, child_prefix))
        return flattened

    shape, dtype = shape_and_dtype(tree, tf_tensor_types)
    path = prefix or "value"
    flattened[path] = ModalitySpec(
        path=path,
        kind=infer_kind(path, tree),
        dtype=dtype,
        shape=shape,
        source="metadata" if prefix else "sample",
    )
    return flattened
