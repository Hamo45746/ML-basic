import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix


def set_seeds(seed_value: int = 1):
    """Sets random seeds"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

def create_tensorboard_writer(
        base_log_dir: str,
        experiment_group: str,
        run_name: str
        ) -> SummaryWriter:
    """Creates a TensorBoard SummaryWriter and log directory.
    
    Args:
        base_log_dir (str): root directory for all runs.
        experiment_group (str): Name for the group of experiments.
        run_name (str): Unique name for this run.
        
    Returns:
        SummaryWriter: TensorBoard SummaryWriter.
    """
    log_dir = os.path.join(base_log_dir, experiment_group, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)  