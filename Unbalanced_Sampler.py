"""
Example script to run attacks in this repository directly without simulation.
This can be useful if you want to check a model architecture and model gradients computed/defended in some shape or form
against some of the attacks implemented in this repository, without implementing your model into the simulation.

All caveats apply. Make sure not to leak any unexpected information.
"""
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import breaching
import neptune
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import sys
import time
from pprint import pprint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from utils.os_utils import Logger, load_config
from libs.datasets import make_dataset, make_data_loader
from utils.torch_utils import fix_random_seed
from models.DeepConvLSTM import DeepConvLSTM
from models.TinyHAR import TinyHAR
from utils.torch_utils import init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from torch.utils.data import DataLoader
from opacus import layers, optimizers

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Custom sampler to choose 50% from class a, 25% from class b, and the last 25% randomly
class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, class_a, class_b):
        self.dataset = dataset
        self.class_a = class_a
        self.class_b = class_b
        self.indices_a = [i for i, (_, label) in enumerate(dataset) if label == class_a]
        self.indices_b = [i for i, (_, label) in enumerate(dataset) if label == class_b]
        self.indices_rest = [i for i, (_, label) in enumerate(dataset) if label != class_a and label != class_b]

    def __iter__(self):
        num_samples = len(self.dataset)
        num_a = num_samples // 2
        num_b = num_samples // 4
        num_rest = num_samples - num_a - num_b

        sampled_indices_a = np.random.choice(self.indices_a, num_a, replace=len(self.indices_a) < num_a)
        sampled_indices_b = np.random.choice(self.indices_b, num_b, replace=len(self.indices_b) < num_b)
        sampled_indices_rest = np.random.choice(self.indices_rest, num_rest, replace=len(self.indices_rest) < num_rest)

        sampled_indices = np.concatenate((sampled_indices_a, sampled_indices_b, sampled_indices_rest))
        np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)

          