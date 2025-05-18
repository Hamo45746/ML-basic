import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import numpy as np
import random
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Optional, List


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
    os.makedirs(log_dir, exist_ok=False) # Don't overwrite existing logs
    print(f"TensorBoard logs will be saved to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)  


def plot_confusion_mat(
        cm: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
        ):
    """Plots and saves the confusion matrix."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
        )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    ax.set_title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    # plt.show()
    plt.close(fig)