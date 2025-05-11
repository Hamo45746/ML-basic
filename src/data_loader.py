import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


def load_data(csv_data: str) -> Optional[pd.DataFrame]:
    """Load the dataset from the csv filepath.
    
    Args:
        csv_data (str): path to the csv file.
    
    Returns:
        pd.DataFrame | None: Loaded dataframe or None.
    """
    try:
        dataframe = pd.read_csv(csv_data, header=0)
        return dataframe
    except Exception as e:
        print(f"Exception loading csv: {e}")
        return None


def select_features(
        dataframe: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """Selects feature and data columns from the pd.dataframe.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe.
        feature_cols (list[str]): List of column names to use as features.
        target_col (str): Name of target column.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: Features dataframe (X) and target 
        series (Y).
    """
    keep_cols = [col for col in feature_cols if col in dataframe.columns]
    features = dataframe[keep_cols]
    target = dataframe[target_col]
    return (features, target)


def split_data(
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: Optional[int] = None,
        stratify: bool = False
        ) -> Tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series,
            pd.Series
            ]:
    """Split data into train, test, validation sets.
    Optionally stratifies the split based on the target var.
    
    Args:
        features (pd.DataFrame): Features.
        target (pd.Series): Target column.
        test_size (float): Fraction of data in test set.
        Validation_size (float): Fraction of data in validation set.
        random_state (Optional[int]): Random seed for reproducibility.
        stratify (bool): Statify split based on target. Reccommended for 
            classification.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) as pandas objects.
    """
    # Input Validation
    if not (0.0 <= test_size <= 1.0):
        raise ValueError(f"test_size not between 0.0 and 1.0, got {test_size}")
    if not (0.0 <= validation_size <= 1.0):
        raise ValueError(f"validation_size not between 0.0 and 1.0, got {validation_size}")
    if test_size + validation_size > 1.0:
        raise ValueError(f"sum of test_size ({test_size}) and validation_size ({validation_size}) above 1.0")

    # Create empty structures
    empty_features_df = pd.DataFrame(columns=features.columns).astype(features.dtypes)
    empty_target_series = pd.Series(dtype=target.dtype, name=target.name)

    stratify_target = target if stratify else None

    # Edge Cases for zero sizes
    if test_size == 1.0:
        return (empty_features_df, empty_features_df, features.copy(),
                empty_target_series, empty_target_series, target.copy())

    if validation_size == 1.0:
        return (empty_features_df, features.copy(), empty_features_df,
                empty_target_series, target.copy(), empty_target_series)

    if test_size + validation_size == 1.0:
        # All data in test and validation
        if test_size == 0.0: # all validation
            return (empty_features_df, features.copy(), empty_features_df,
                    empty_target_series, target.copy(), empty_target_series)

        try:
            X_val, X_test, y_val, y_test = train_test_split(
                features,
                target,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_target
                )
        except ValueError as e:
            raise ValueError(f"Stratification failed: {e}") from e

        return (empty_features_df, X_val, X_test,
                empty_target_series, y_val, y_test)

    # Standard Splitting (Train + Val + Test > 0
    if test_size == 0.0:
        X_train_val, X_test = features.copy(), empty_features_df
        y_train_val, y_test = target.copy(), empty_target_series
    else:
        # Separate Test set
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_target
                )
        except ValueError as e:
            raise ValueError(f"Stratification failed - train/test split: {e}") from e

    if validation_size == 0.0:
        X_train, X_val = X_train_val, empty_features_df
        y_train, y_val = y_train_val, empty_target_series
    else:
        # Split val set from train_val pool
        # Calc relative val size in train_val pool
        original_train_val_prop = 1.0 - test_size
        relative_val_size = validation_size / original_train_val_prop

        # Check if train_val set is empty (can happen with tiny datasets)
        if X_train_val.empty:
            X_train, X_val = empty_features_df, empty_features_df
            y_train, y_val = empty_target_series, empty_target_series
        else:
            stratify_target_val = y_train_val if stratify else None
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=relative_val_size,
                    random_state=random_state,
                    shuffle=True,
                    stratify=stratify_target_val
                    )
            except ValueError as e:
                raise ValueError(f"Stratification failed in train/validation split: {e}") from e

    return X_train, X_val, X_test, y_train, y_val, y_test