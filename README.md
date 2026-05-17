# rldata

A Python package for loading robot learning datasets, with support for the [Open X-Embodiment (OXE)](./doc/OXE.md) collection.

## Installation

```bash
pip install "rldata[oxe]"
```

Or from source:

```bash
git clone https://github.com/robotics-action-group/rldata.git
cd rldata
pip install -e ".[oxe]"
```

**Requirements:** Python >= 3.7, PyTorch >= 1.13.1, TensorFlow >= 2.11.1, TensorFlow Datasets >= 4.8.2

## Quick Start

```python
from rldata import OXEDataset, list_datasets

dataset = OXEDataset(dataset_name='droid', split='train')
datasets = list_datasets()

print(len(dataset))
print(dataset.num_episodes)
print(dataset.get_modalities())
print(dataset.get_dataset_info())

for sample in dataset:
    print(sample.keys())
    break
```

## OXEDataset

`OXEDataset` is a TorchRL `BaseDatasetExperienceReplay` for Open
X-Embodiment datasets in `gs://gresearch/robotics`.

### Discover datasets and versions

```python
from rldata import list_datasets, validate_dataset_name, dataset2path

datasets = list_datasets()
print(datasets["viola"])  # example: ['0.1.0']

print(validate_dataset_name("droid"))        # True/False
print(dataset2path("droid"))                 # latest version path
print(dataset2path("droid", version="1.0.1"))  # explicit version path
```

- `list_datasets()` returns `dict[str, list[str]]` with versions sorted newest-first.
- If `version` is omitted, the latest available version is selected automatically.

### Constructor

```python
OXEDataset(
    dataset_name: str = "droid",
    split: str = "train",
    version: str | None = None,
    episodes: list[int] | None = None,
    batch_size: int = 32,
    slice_len: int | None = None,
    root: str | None = None,
)
```

### Download behavior

1. **Full dataset mode** (`episodes=None`)
   - downloads metadata + shards into local cache,
   - converts all episodes for the split.

2. **Episode-selective mode** (`episodes=[...]`)
   - downloads metadata only,
   - streams and converts only requested missing episodes.

Progress bars are shown when `tqdm` is installed.

### Cache behavior

Cache root priority:

1. `root` argument
2. `RLDATA_CACHE` environment variable
3. `~/.cache/rldata` (default)

The cache is incremental:

- already-converted episodes are reused,
- missing episodes are converted once,
- combined memmap storage is reused across repeated loads.

### Data format and metadata

Each step is stored in TorchRL TED-compatible structure with keys:

- `observation`
- `action`
- `done`
- `terminated`
- `next` (`observation`, `reward`, `done`, `terminated`)
- `collector.traj_ids`

Public API helpers:

- `num_episodes`: number of loaded episodes.
- `get_modalities()`: flattened modality specs inferred from TFDS metadata.
- `get_dataset_info()`: dataset description/features/splits from TFDS info.
- `data_path`: per-episode TED cache directory.
- `data_path_root`: root cache path for this dataset/version.

## License

MIT — see [LICENSE](LICENSE).
