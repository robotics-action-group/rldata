"""RLData - A Python package for robot learning dataset handling.

This package provides utilities for loading and handling robot learning datasets,
with support for the OXE (Open X-Embodiment) dataset collection from Google Cloud.
"""

from rldata.oxe_dataset import (
    OXEDataset,
    dataset2path,
    list_datasets,
    validate_dataset_name,
)

__all__ = [
    'OXEDataset',
    'dataset2path',
    'list_datasets',
    'validate_dataset_name',
]

__version__ = '0.1.0'
__author__ = 'Robotics Action Group'
