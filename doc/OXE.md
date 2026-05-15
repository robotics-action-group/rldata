# OXEDataset

`OXEDataset` is a TorchRL `BaseDatasetExperienceReplay` implementation for the
[Open X-Embodiment](https://robotics-transformer-x.github.io/) datasets hosted
under `gs://gresearch/robotics`.

It handles dataset discovery, download, conversion to TorchRL TED format, and
memory-mapped caching for fast reuse.

## Import

```python
from rldata import OXEDataset, list_datasets, dataset2path, validate_dataset_name
```

## Discover available datasets and versions

Use `list_datasets()` to inspect the OXE bucket:

```python
datasets = list_datasets()
print(datasets["viola"])  # example: ['0.1.0']
```

- Returns `dict[str, list[str]]`.
- Versions are sorted newest-first.

Helper APIs:

- `validate_dataset_name(dataset_name, version=None) -> bool`
- `dataset2path(dataset_name, version=None) -> str`
  - When `version=None`, the latest available version is selected.

## Basic usage

```python
dataset = OXEDataset(dataset_name="droid", split="train")
batch = dataset.sample()

print(len(dataset))            # number of steps
print(dataset.num_episodes)    # number of loaded episodes
print(dataset.get_modalities())
```

## Constructor

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

### Parameters

- `dataset_name`: OXE dataset name in the bucket.
- `split`: TFDS split (for example `"train"`).
- `version`: explicit dataset version; latest version is used when omitted.
- `episodes`: optional list of episode indices to load.
- `batch_size`: number of transitions returned by `sample()`.
- `slice_len`: if set, samples contiguous trajectory slices via `SliceSampler`.
- `root`: optional cache root override.

## Download modes

`OXEDataset` supports two loading strategies:

1. **Full dataset mode** (`episodes=None`)
   - Downloads metadata and shard files to local cache.
   - Converts all episodes for the selected split.

2. **Episode-selective mode** (`episodes=[...]`)
   - Downloads metadata only.
   - Streams only requested missing episodes during conversion.
   - Avoids downloading full shard sets when not needed.

Both modes show progress bars when `tqdm` is available.

## Cache behavior

Cache root priority:

1. `root` constructor argument
2. `RLDATA_CACHE` environment variable
3. `~/.cache/rldata` (default)

The loader is incremental:

- already-converted episodes are reused,
- missing episodes are built only once,
- combined memmap storage is reused for repeated episode selections.

## Data format

Converted data is stored as TorchRL TED-compatible transitions:

- `observation`
- `action`
- `done`
- `terminated`
- `next` (contains next observation/reward/done/terminated)
- `collector.traj_ids` (trajectory ids for slice sampling)

Storage uses `TensorDict` memmaps through `TensorStorage`, so training reads are
lazy and memory-efficient.

## Metadata and modalities

- `get_dataset_info()` exposes TFDS builder metadata (description/features/splits).
- `get_modalities()` returns flattened modality specs inferred from `builder.meta`
  (or fallback features), allowing different OXE datasets to expose different
  modality structures safely.

## Public properties

- `num_episodes`: number of loaded episodes.
- `data_path`: per-episode TED cache directory for the split.
- `data_path_root`: root cache path for the selected dataset/version.
