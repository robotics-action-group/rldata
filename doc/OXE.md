We need a class OXEDataset(torch.utils.data.Dataset) that can load data for specific dataset part of the oxe gcloud bucket.

REMEMBER AGENT.md is the entrypoint for the agent.

use the file oxe_dataset.py

1. List all datatsets in the oxe gcloud bucket, gs://gresearch/robotics
2. When initializing the OXEDataset, we should be able to specify which dataset to load. It can load tensorflow Dataset.