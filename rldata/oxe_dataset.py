import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import imageio
from IPython.display import Video, Image

# Set default dtype for torch
# torch.set_default_dtype(torch.bfloat16)
torch.set_default_dtype(torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset2path(dataset_name: str) -> str:
    """Convert dataset name to its GCS path.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'robo_net', 'language_table', 'droid')
        
    Returns:
        Google Cloud Storage path to the dataset
    """
    # Dataset version mapping
    versions = {
        'robo_net': '1.0.0',
        'language_table': '0.0.1',
        'droid': '1.0.1',
    }
    version = versions.get(dataset_name, '0.1.0')
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


class OXEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for OXE (Open X-Embodiment) datasets from Google Cloud.
    
    Loads tensorflow datasets from the gs://gresearch/robotics bucket and provides
    a PyTorch-compatible interface.
    
    Attributes:
        dataset_name: Name of the dataset to load
        split: Dataset split (e.g., 'train', 'validation', 'test')
        builder: TensorFlow dataset builder
        dataset: TensorFlow dataset object
        samples: List of samples from the dataset
    """
    
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
            split: Dataset split to load (default: 'train')
            shuffle_files: Whether to shuffle files (default: False)
            **kwargs: Additional arguments passed to as_dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle_files = shuffle_files
        
        # Get the path to the dataset
        builder_dir = dataset2path(dataset_name)
        
        try:
            # Create builder from directory
            self.builder = tfds.builder_from_directory(builder_dir=builder_dir)
            
            # Load the dataset
            self.dataset = self.builder.as_dataset(
                split=split,
                shuffle_files=shuffle_files,
                **kwargs
            )
            
            # Get dataset info
            self.info = self.builder.info
            
            # Convert to list for indexing (may use cache in future for memory efficiency)
            self.samples = list(self.dataset)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from {builder_dir}. "
                f"Error: {str(e)}"
            )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the sample data
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.samples)})")
        
        sample = self.samples[idx]
        
        # Convert TensorFlow tensors to PyTorch tensors if needed
        if isinstance(sample, dict):
            torch_sample = {}
            for key, value in sample.items():
                if isinstance(value, (np.ndarray, tf.Tensor)):
                    torch_sample[key] = torch.from_numpy(
                        np.array(value) if isinstance(value, tf.Tensor) else value
                    )
                else:
                    torch_sample[key] = value
            return torch_sample
        
        return sample
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        if self.info:
            return {
                'name': self.dataset_name,
                'split': self.split,
                'num_samples': len(self),
                'features': self.info.features if hasattr(self.info, 'features') else None,
                'description': self.info.description if hasattr(self.info, 'description') else None,
            }
        return {'name': self.dataset_name, 'split': self.split, 'num_samples': len(self)}


# Add tensorflow import at the end to avoid issues
try:
    import tensorflow as tf
except ImportError:
    tf = None