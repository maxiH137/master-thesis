"""Repeatable code parts concerning data loading.
Data Config Structure (cfg_data): See config/data
"""


import torch

from .cached_dataset import CachedDataset
from Samplers import UnbalancedSampler, BalancedSampler # Added for HAR
import random # Added for HAR
import torch.backends.cudnn as cudnn
import numpy as np
import os
# Block ImageNet corrupt EXIF warnings
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def construct_dataloader(cfg_data, cfg_impl, user_idx=0, return_full_dataset=False, dataset=None):
    """Return a dataloader with given dataset for the given user_idx.

    Use return_full_dataset=True to return the full dataset instead (for example for analysis).
    """
    if cfg_data.modality == "vision":
        from .datasets_vision import _build_dataset_vision, _split_dataset_vision

        dataset, collate_fn = _build_dataset_vision(cfg_data, split=cfg_data.examples_from_split, can_download=True)
        dataset = _split_dataset_vision(dataset, cfg_data, user_idx, return_full_dataset)
    elif cfg_data.modality == "text":
        from .datasets_text import _build_and_split_dataset_text

        dataset, collate_fn = _build_and_split_dataset_text(
            cfg_data, cfg_data.examples_from_split, user_idx, return_full_dataset,
        )
    elif cfg_data.modality == "inertial":
        from .datasets_har import _build_dataset_har, _split_dataset_har
        from .datasets_har import collate_function
        collate_fn = collate_function
        #dataset, collate_fn = _build_dataset_har(cfg_data, split=cfg_data.examples_from_split, can_download=True)
        #dataset = _split_dataset_har(dataset, cfg_data, user_idx, return_full_dataset)
    else:
        raise ValueError(f"Unknown data modality {cfg_data.modality}.")

    if len(dataset) == 0:
        raise ValueError("This user would have no data under the chosen partition, user id and number of clients.")

    if cfg_data.db.name == "LMDB":
        from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb, that's why it's a lazy import

        dataset = LMDBDataset(dataset, cfg_data, cfg_data.examples_from_split, can_create=True)

    if cfg_data.caching:
        dataset = CachedDataset(dataset, num_workers=cfg_impl.threads, pin_memory=cfg_impl.pin_memory)

    if cfg_impl.threads > 0:
        num_workers = (
            min(torch.get_num_threads(), cfg_impl.threads * max(1, torch.cuda.device_count()))
            if torch.get_num_threads() > 1
            else 0
        )
    else:
        num_workers = 0

    if cfg_impl.shuffle:
        data_sampler = torch.utils.data.RandomSampler(dataset, replacement=cfg_impl.sample_with_replacement)
    else:
        data_sampler = torch.utils.data.SequentialSampler(dataset)
           
    # Changed to enable Shuffle in HAR    
    if cfg_data.name == "Wear":
        classes = 18
    else: 
        classes = 8
        
    unbalanced_sampler = UnbalancedSampler(dataset, random.randint(0, classes), random.randint(0, classes))
    balanced_sampler = BalancedSampler(dataset)
    rng_generator = fix_random_seed(1, include_cuda=True) 
    
    if cfg_data.shuffle == "shuffle": 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(cfg_data.batch_size, len(dataset)),
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            generator = rng_generator,
        )
    elif cfg_data.shuffle == "sequential":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(cfg_data.batch_size, len(dataset)),
            sampler=data_sampler,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            generator = rng_generator,
        )
    elif cfg_data.shuffle == "balanced": 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(cfg_data.batch_size, len(dataset)),
            sampler = balanced_sampler,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            generator = rng_generator,
        )
    elif cfg_data.shuffle == "unbalanced":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(cfg_data.batch_size, len(dataset)),
            sampler=unbalanced_sampler,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=cfg_impl.pin_memory,
            persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
            generator = rng_generator,
        )    
    
    # Save the name for later:
    dataloader.name = cfg_data.name

    return dataloader
