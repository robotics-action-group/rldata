We need a class OXEDataset(torch.utils.data.IterativeDataset) that can load data for specific dataset part of the oxe gcloud bucket.

REMEMBER AGENT.md is the entrypoint for the agent.

use the file oxe_dataset.py, make more files if needed.

List of features:

1. List all datatsets with their available versions in the oxe gcloud bucket, gs://gresearch/robotics. A function that lists all datasets in the oxe gcloud bucket, gs://gresearch/robotics. This can be a helper function that is used by OXEDataset to validate the dataset name or version and get the path to the dataset.
For example in output I want to see, name: viola and versions [0.1.0] 

2. When initializing the OXEDataset, we should be able to specify which dataset to load. It can load tensorflow Dataset. The version is automatically loaded with highest number.

3. Give multiple options for downloading the dataset from internet, such as downloading the entire dataset to local storage, or downloading only few episodes through episodes parameter which takes a list of episode indices. If episodes list is given download that subset otherwise download full. When downloading show progress bar, like how LeRobotDataset does it. Feel free to imitate this part from LeRobotDataset.

4. Download the dataset into a local cache directory, so that it can be reused without downloading again. The default cache directory can be ~/.cache/rldata but it should be configurable through an environment variable RLDATA_CACHE.

5. After Downloading, use builder.meta to get all the modalities for in any episode. Remember, modalities change from datatset to dataset, so we need to be able to handle that.

6. tf tensor to torch tensor conversion. Do it only at the __iter__ method. First convert tf tensor to numpy and then to torch tensor. Remeber it has to efficient and memory safe. Do not convert the entire dataset to torch tensor at once, do it on the fly when iterating through the dataset. Dont use or move to cuda or gpu device anywhere in the Dataset.