import torch
import os

import imageio
import yaml
import torchvision
from skimage.transform import resize
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)

import numpy as np
import random 
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import shutil
# from mpi4py import MPI
import nibabel as nib
import pickle5 as pickle

from filelock import FileLock

default_num_worker = 4

class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)




def data_loader(
    batch_size,
    drop_last = True,
    shuffle = True,
    sampler = None,
    split_val = False
):


    train_loader = torch.utils.data.Dataloader(
        train_dataset,
        num_worker = default_num_worker,
        sampler = sampler,
        shuffle = shuffle,
        batch_size = batch_size
        drop_last = drop_last
    )

    val_loader = torch.utils.data.Dataloader(
        val_dataset,
        num_worker = default_num_worker,
        sampler = sampler,
        shuffle = shuffle,
        batch_size = batch_size
        drop_last = drop_last
    )

    return train_loader



if __name__ == "main":
    data_loader()