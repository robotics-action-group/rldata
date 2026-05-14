from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rldata.oxe_dataset as oxe
from rldata.oxe.utils import tf_to_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CACHE = {"droid": {"1.0.1": "gs://gresearch/robotics/droid/1.0.1"}}

_EPISODES = [
    {
        "observation": {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "state": np.array([float(i)] * 4, dtype=np.float32),
        },
        "action": np.array([float(i), float(i)], dtype=np.float32),
        "language_instruction": b"pick up the block",
    }
    for i in range(5)
]


class FakeDataset:
    def __init__(self, items):
        self._items = list(items)

    def take(self, n):
        return FakeDataset(self._items[:n])

    def skip(self, n):
        return FakeDataset(self._items[n:])

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakeInfo:
    description = "fake"

    def __init__(self, episodes):
        self.features = {
            "observation": {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "state": np.zeros((4,), dtype=np.float32),
            },
            "action": np.zeros((2,), dtype=np.float32),
            "language_instruction": b"pick up the block",
        }
        self.splits = {"train": SimpleNamespace(num_examples=len(episodes))}


class FakeBuilder:
    def __init__(self, episodes):
        self.info = FakeInfo(episodes)
        self.meta = self.info.features
        self._episodes = episodes

    def as_dataset(self, split, shuffle_files=False, **kwargs):
        return FakeDataset(self._episodes)


def _patch(monkeypatch: pytest.MonkeyPatch, episodes=None):
    eps = episodes if episodes is not None else _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda builder_dir: FakeBuilder(eps))
    monkeypatch.setattr(oxe, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)
    # Bypass GCS download; builder_from_directory is already mocked above
    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: None)


# ---------------------------------------------------------------------------
# tf_to_torch — conversion coverage
# ---------------------------------------------------------------------------

def test_tf_to_torch_numeric_array() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = tf_to_torch(arr)
    assert isinstance(out, torch.Tensor)
    assert out.tolist() == pytest.approx([1.0, 2.0])


def test_tf_to_torch_uint8_image() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = tf_to_torch(img)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.uint8


def test_tf_to_torch_python_scalars() -> None:
    assert isinstance(tf_to_torch(3), torch.Tensor)
    assert isinstance(tf_to_torch(3.14), torch.Tensor)
    assert isinstance(tf_to_torch(True), torch.Tensor)


def test_tf_to_torch_numpy_scalar() -> None:
    out = tf_to_torch(np.float32(2.5))
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(2.5)


def test_tf_to_torch_bytes_decoded_to_str() -> None:
    out = tf_to_torch(b"pick up the block")
    assert isinstance(out, str)
    assert out == "pick up the block"


def test_tf_to_torch_nested_dict() -> None:
    data = {"obs": np.array([1.0], dtype=np.float32), "label": b"go"}
    out = tf_to_torch(data)
    assert isinstance(out["obs"], torch.Tensor)
    assert isinstance(out["label"], str)


# ---------------------------------------------------------------------------
# list_datasets / validate / dataset2path
# ---------------------------------------------------------------------------

def test_list_datasets_returns_version_list_per_dataset() -> None:
    datasets = oxe.list_datasets()
    assert isinstance(datasets, dict)
    for name, versions in datasets.items():
        assert isinstance(versions, list)
        assert all(isinstance(v, str) for v in versions)
    if "droid" in datasets:
        assert "1.0.1" in datasets["droid"]


def test_dataset_path_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)
    assert oxe.validate_dataset_name("droid") is True
    assert oxe.validate_dataset_name("missing-dataset") is False
    assert oxe.dataset2path("droid") == "gs://gresearch/robotics/droid/1.0.1"


# ---------------------------------------------------------------------------
# Cache dir
# ---------------------------------------------------------------------------

def test_get_cache_dir_default() -> None:
    path = oxe._get_cache_dir()
    assert str(path).endswith(".cache/rldata")


def test_get_cache_dir_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLDATA_CACHE", str(tmp_path / "custom_cache"))
    path = oxe._get_cache_dir()
    assert path == tmp_path / "custom_cache"


def test_get_cache_dir_override(tmp_path: Path) -> None:
    path = oxe._get_cache_dir(override=str(tmp_path / "explicit"))
    assert path == tmp_path / "explicit"


# ---------------------------------------------------------------------------
# OXEDataset — basic construction and iteration
# ---------------------------------------------------------------------------

def test_strip_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid/", split="train")
    assert ds.dataset_name == "droid"


def test_iter_converts_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train")
    sample = next(iter(ds))
    # Numeric arrays → torch.Tensor
    assert isinstance(sample["action"], torch.Tensor)
    assert sample["action"].shape == (2,)
    assert isinstance(sample["observation"]["image"], torch.Tensor)
    assert isinstance(sample["observation"]["state"], torch.Tensor)
    # Byte string → decoded str (cannot be a tensor)
    assert isinstance(sample["language_instruction"], str)


def test_iter_all_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train")
    results = list(ds)
    assert len(results) == 5
    for i, ep in enumerate(results):
        assert ep["action"][0].item() == pytest.approx(float(i))


def test_getitem(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train")
    ep = ds[2]
    assert isinstance(ep["action"], torch.Tensor)
    assert ep["action"][0].item() == pytest.approx(2.0)

    with pytest.raises(IndexError):
        ds[99]


# ---------------------------------------------------------------------------
# OXEDataset — episodes filter
# ---------------------------------------------------------------------------

def test_episodes_filters_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 2, 4])
    results = list(ds)
    assert len(results) == 3
    assert results[0]["action"][0].item() == pytest.approx(0.0)
    assert results[1]["action"][0].item() == pytest.approx(2.0)
    assert results[2]["action"][0].item() == pytest.approx(4.0)


def test_episodes_single(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[3])
    results = list(ds)
    assert len(results) == 1
    assert results[0]["action"][0].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# OXEDataset — local cache
# ---------------------------------------------------------------------------

def test_local_tfds_dir_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert ds._local_tfds_dir() == tmp_path / "oxe" / "droid"


def test_local_tfds_dir_with_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", version="1.0.1", root=str(tmp_path))
    assert ds._local_tfds_dir() == tmp_path / "oxe" / "droid" / "1.0.1"


def test_copy_tree_called_when_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    eps = _EPISODES
    fake_tf = SimpleNamespace(Tensor=np.ndarray, io=SimpleNamespace(gfile=SimpleNamespace()))
    fake_tfds = SimpleNamespace(builder_from_directory=lambda builder_dir: FakeBuilder(eps))
    monkeypatch.setattr(oxe, "tf", fake_tf, raising=False)
    monkeypatch.setattr(oxe, "tfds", fake_tfds, raising=False)
    monkeypatch.setattr(oxe, "_TF_TENSOR_TYPES", tuple(), raising=False)
    monkeypatch.setattr(oxe, "_DATASET_CACHE", dict(_FAKE_CACHE), raising=False)

    calls = []
    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: calls.append((src, dst)))

    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert len(calls) == 1
    assert calls[0][0] == "gs://gresearch/robotics/droid/1.0.1"


def test_copy_tree_skipped_when_already_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    # Pre-create the local TFDS dir with a valid (non-empty) dataset_info.json
    local_dir = tmp_path / "oxe" / "droid"
    local_dir.mkdir(parents=True)
    (local_dir / "dataset_info.json").write_text("{}")

    calls = []
    monkeypatch.setattr(oxe, "_copy_tree", lambda src, dst: calls.append((src, dst)))

    oxe.OXEDataset(dataset_name="droid", split="train", root=str(tmp_path))
    assert len(calls) == 0


# ---------------------------------------------------------------------------
# Modalities and dataset info
# ---------------------------------------------------------------------------

def test_modalities_from_builder_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train")
    mods = ds.get_modalities()
    assert mods["observation/image"]["kind"] == "image"
    assert mods["action"]["kind"] == "action"


def test_get_dataset_info(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    ds = oxe.OXEDataset(dataset_name="droid", split="train")
    info = ds.get_dataset_info()
    assert info["description"] == "fake"
