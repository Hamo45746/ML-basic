import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.cuda.amp import GradScaler, autocast#Unsure if use
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from typing import Tuple
import numpy as np


def select_device() -> torch.device:
    """Use Cuda if possible, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def define_optimiser(
        model: nn.Module,
        optimiser_name: str = 'adam',
        lr: float = 0.001
        ) -> optim.Optimizer | None:
    """Create optimiser - Adam or SGD."""
    if optimiser_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimiser_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimiser: optim.Optimizer,
        device: torch.device
        ) -> tuple[float, float]:
    """Performs single training epoch.
    Assumes model already on device.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for training data.
        loss_fn (nn.Module): The loss function.
        optimiser (optim.Optimizer): The optimiser.
        device (torch.device): Device to train on.

    Returns:
        tuple[float, float]: Average training loss, accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

    
def eval_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
        ) -> tuple[float, float]:
    """Performs validation on a set for one epoch.
    Assumes model already on device.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for validation data.
        loss_fn (nn.Module): The loss function.
        device (torch.device): Device to train on.

    Returns:
        tuple[float, float]: Average training loss, accuracy.
    """
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    return (val_loss / len(dataloader), 100. * val_correct / val_total)


def run_training_loop(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimiser: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        writer: SummaryWriter
        ) -> None:
    """Runs main training loop for epochs, including validation.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimiser (optim.Optimizer): The optimiser.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device to use.
        epochs (int): Num epochs to train.
        writer (SummaryWriter): TensorBoard writer.
    """
    model = model.to(device)

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model,
                                           train_loader,
                                           loss_fn,
                                           optimiser,
                                           device
                                           )
        val_loss, val_acc = eval_step(model,
                                      val_loader,
                                      loss_fn,
                                      device
                                      )
        print(f'Epoch: {epoch}; Train Loss: {train_loss}; \
            Train Accuracy: {train_acc}')

        print(f'Epoch: {epoch}; Val Loss: {val_loss}; \
            Val Accuracy: {val_acc}')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', val_acc, epoch)
        
        
def test_set_eval(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model on test dataloader.
    returns loss, accuracy, predictions and labels."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    correct_samples = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            actual_labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, actual_labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(actual_labels.cpu().numpy())
            
            correct_samples += (preds == actual_labels).sum().item()
            total_samples += actual_labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = (correct_samples / total_samples) * 100.0 if total_samples > 0 else 0
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)