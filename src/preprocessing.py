import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


def normalise_features(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
        ) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            MinMaxScaler
            ]:
    """Applies Min-Max scaling to the feature sets - for the training data.
    
    Args:
        X_train, X_val, X_test (pd.DataFrame): Feature DataFrame per split.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: Scaled feature
        arrays (NumPy) for train, val, test, and fitted scaler object.
    """
    # Will create normalised copy, could use in-place update instead?
    scaler = MinMaxScaler()
    if X_train.empty:
        print("X_train empty. Scaler cannot fit. Returning unscaled data.")
        return (X_train.to_numpy(), X_val.to_numpy(),
                X_test.to_numpy(), scaler)

    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    # transform non-empty dataframes
    X_val_norm = (
            scaler.transform(X_val)
            if not X_val.empty
            else np.array([]).reshape(0, X_train.shape[1])
            )
    X_test_norm = (
            scaler.transform(X_test)
            if not X_test.empty
            else np.array([]).reshape(0, X_train.shape[1])
            )

    return X_train_norm, X_val_norm, X_test_norm, scaler


class TTSwingDataset(Dataset):
    """PyTorch Dataset for the swing data."""
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Init function.

        Args:
            features (np.ndarray): The input features.
            labels (np.ndarray): Matching labels.
        """
        super().__init__()
        try:
            self.features_tensor = torch.from_numpy(features).float()
            self.targets_tensor = torch.from_numpy(labels).float() 
        except TypeError as e:
             print(f"Error converting numpy arrays to tensors. \
             Features dtype: {features.dtype}, Labels dtype: {labels.dtype}")
             raise e

        # dimension check
        if self.features_tensor.shape[0] != self.targets_tensor.shape[0]:
            raise ValueError(f"Num samples in features \
            ({self.features_tensor.shape[0]}) does not match labels \
            ({self.targets_tensor.shape[0]})")
 
    def __len__(self) -> int:
        """Returns total sample count."""
        return len(self.features_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets sample (features, label) at given index."""
        return (self.features_tensor[idx], self.targets_tensor[idx])


def create_dataloader(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        batch_size: int
        ) -> Tuple[
            Optional[DataLoader],
            Optional[DataLoader],
            Optional[DataLoader]
            ]:
    """Creates PyTorch DataLoaders for training, validation, and test sets.

    Args:
        X_train, y_train, etc. (np.ndarray): Preprocessed features and 
            labels for each split.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train,
        validation, and test sets.
    """
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    if X_train.size > 0: 
        train_dataset = TTSwingDataset(features=X_train, labels=y_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 # Set num_workers later
            )
    else:
        print("Warning: Training data empty. No DataLoader created.")

    if X_val.size > 0:
        val_dataset = TTSwingDataset(features=X_val, labels=y_val)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
            )
    else:
        print("Warning: Validation data empty. DataLoader not created.")

    if X_test.size > 0:
        test_dataset = TTSwingDataset(features=X_test, labels=y_test)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
            )
    else:
         print("Warning: Test data empty. DataLoader not created.")

    return (train_dataloader, val_dataloader, test_dataloader)
