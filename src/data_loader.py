import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


def load_data(csv_data: str) -> Optional[pd.DataFrame]:
    """Load the dataset from the csv filepath."""
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
    """Selects feature and data columns from pd.dataframe."""
    keep_cols = [col for col in feature_cols if col in dataframe.columns]
    if len(keep_cols) != len(feature_cols):
        print(f"Some feature cols not found")
        print(f"Requested: {feature_cols}, Found: {keep_cols}")
    
    features = dataframe[keep_cols]
    target = dataframe[target_col]
    return features, target


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    validation_ratio: float,
    test_ratio: float,
    random_state: Optional[int] = None
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series
        ]:
    """
    Splits data into train, validation, and test sets.
    Always stratifies.

    Assumptions:
    1. 0 < validation_ratio < 1
    2. 0 < test_ratio < 1
    3. validation_ratio + test_ratio < 1.0
    """
    temp_pool_ratio = validation_ratio + test_ratio

    X_train, X_temp_pool, y_train, y_temp_pool = train_test_split(
        features,
        target,
        test_size=temp_pool_ratio,
        random_state=random_state,
        stratify=target,
        shuffle=True
        )

    proportion_for_test = test_ratio / temp_pool_ratio

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp_pool,
        y_temp_pool,
        test_size=proportion_for_test,
        random_state=random_state,
        stratify=y_temp_pool,
        shuffle=True
        )

    return X_train, X_val, X_test, y_train, y_val, y_test