# rldata

A Python package for loading and handling robot learning datasets, with support for the OXE (Open X-Embodiment) dataset collection from Google Cloud Storage.

## Overview

`rldata` provides a unified interface for accessing robot learning datasets, particularly those hosted in the Open X-Embodiment (OXE) collection on Google Cloud Storage. It simplifies loading large-scale multi-robot datasets into PyTorch for training and analysis.

## Features

- **OXE Dataset Support**: Load datasets from the `gs://gresearch/robotics` bucket
- **PyTorch Integration**: Native PyTorch `Dataset` interface for easy integration with training pipelines
- **TensorFlow Dataset Support**: Leverages TensorFlow Datasets (tfds) for robust data loading
- **Multi-Dataset Support**: Easy switching between different robot learning datasets
- **Dataset Metadata**: Access dataset information and schema through the API

## Installation

### Using pip

```bash
pip install "rldata[oxe]"
```

### From source

```bash
git clone https://github.com/robotics-action-group/rldata.git
cd rldata
pip install -e ".[oxe]"
```

### Requirements

- Python >= 3.7
- PyTorch >= 1.13.1
- NumPy
- TensorFlow >= 2.11.1 and TensorFlow Datasets >= 4.8.2 (required for OXE loading)

## Quick Start

### Loading an OXE Dataset

```python
from rldata import OXEDataset

# Load the DROID dataset from OXE
dataset = OXEDataset(dataset_name='droid', split='train')

# Access dataset metadata
print(f"Dataset size: {len(dataset)}")
print(f"Dataset info: {dataset.get_dataset_info()}")

# Iterate through samples
for i, sample in enumerate(dataset):
    if i >= 10:
        break
    print(f"Sample {i}: {sample.keys()}")
```

### Available Datasets

The following datasets are available in the OXE collection:

- `droid`: DROID (Distributed Robot Interaction Open Dataset)
- `robo_net`: RoboNet dataset
- `language_table`: Language Table dataset
- And more...

## Usage

### Basic Dataset Loading

```python
from rldata import OXEDataset

# Initialize dataset
dataset = OXEDataset(
    dataset_name='droid',
    split='train',
    shuffle_files=False
)

# Get dataset size
print(len(dataset))

# Access a single sample
sample = dataset[0]
print(sample.keys())  # See available fields
```

### Working with PyTorch DataLoader

```python
import torch
from torch.utils.data import DataLoader
from rldata import OXEDataset

# Create dataset
dataset = OXEDataset(dataset_name='droid', split='train')

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch in dataloader:
    # Process batch
    pass
```

## API Reference

### OXEDataset

```python
class OXEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for OXE datasets from Google Cloud."""
    
    def __init__(
        self,
        dataset_name: str = 'droid',
        split: str = 'train',
        shuffle_files: bool = False,
        **kwargs
    ):
        """Initialize OXE Dataset.
        
        Args:
            dataset_name: Name of the dataset to load from OXE bucket
            split: Dataset split ('train', 'validation', 'test')
            shuffle_files: Whether to shuffle files
            **kwargs: Additional arguments for as_dataset()
        """
```

### dataset2path

```python
def dataset2path(dataset_name: str) -> str:
    """Convert dataset name to its GCS path.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Google Cloud Storage path to the dataset
    """
```

## Documentation

For more information, see the [OXE documentation](./doc/OXE.md).

## Jupyter Notebooks

Example Jupyter notebooks demonstrating usage:

- `data_load_oxe.ipynb`: Loading OXE datasets
- `data_load_tfds.ipynb`: Loading TensorFlow datasets directly
- `obs_only_viz.ipynb`: Visualization examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rldata2024,
  title={RLData: Robot Learning Dataset Utilities},
  author={Robotics Action Group},
  year={2024},
  url={https://github.com/robotics-action-group/rldata}
}
```

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/robotics-action-group/rldata).
