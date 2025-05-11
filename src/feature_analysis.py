import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


DATA_FILE_PATH = "./data/TTSwing.csv" # Assumes running from repo folder
TARGET_COLUMN = 'testmode'
N_SPLITS_CV = 5 # Number of folds
RANDOM_STATE_BASE = 0

# Define potential features
POTENTIAL_FEATURES = [
    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
    'a_mean', 'a_max', 'a_min', 'a_skewn', 'a_kurt',
    'g_mean', 'g_max', 'g_min', 'g_skewn', 'g_kurt',
    'a_fft', 'g_fft', 'a_psdx', 'g_psdx', 'a_entropy', 'g_entropy'
    ]


def main():
    # Load Data
    dataframe = pd.read_csv(DATA_FILE_PATH)
    # Drop rows if NaN
    dataframe.dropna(
        subset=[TARGET_COLUMN] + POTENTIAL_FEATURES,
        inplace=True
        )
    
    y = dataframe[TARGET_COLUMN].astype(int)
    X = dataframe[POTENTIAL_FEATURES].copy()
    
    if X.empty or len(y) == 0:
        print("No non-NaN data somehow")
        return
    
    all_fold_importances = []
    fold_val_accuracies = []
    fold_train_accuracies = []
    
    # StratifiedKFold because unbalanced class numbers 
    skf = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True, 
        random_state=RANDOM_STATE_BASE
        )
    
    fold_num = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\nFold {fold_num}/{N_SPLITS_CV}")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        forest = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE_BASE + fold_num,
            class_weight='balanced'
            )
        
        forest.fit(X_train_fold, y_train_fold)
        
        all_fold_importances.append(forest.feature_importances_)
        
        # Evaluate on folds train set
        y_pred_train_fold = forest.predict(X_train_fold)
        accuracy_train_fold = accuracy_score(y_train_fold, y_pred_train_fold)
        fold_train_accuracies.append(accuracy_train_fold)
        
        # Evaluate on folds val set
        y_pred_val_fold = forest.predict(X_val_fold)
        accuracy_val_fold = accuracy_score(y_val_fold, y_pred_val_fold)
        fold_val_accuracies.append(accuracy_val_fold)
        
        print(f"Fold {fold_num} Train Accuracy: {accuracy_train_fold:.4f}")
        print(f"Fold {fold_num} Validation Accuracy: {accuracy_val_fold:.4f}")
        fold_num += 1
        
    average_importances_cv = np.mean(all_fold_importances, axis=0)
    std_importances_cv = np.std(all_fold_importances, axis=0)
    
    feature_importances_df = pd.DataFrame({
        'feature': POTENTIAL_FEATURES,
        'average_importance_cv': average_importances_cv,
        'std_importance_cv': std_importances_cv
        })

    features_sorted_cv = feature_importances_df.sort_values(
        by='average_importance_cv',
        ascending=False
        )

    print("\nAvg Feature Importances")
    print(features_sorted_cv[[
        'feature',
        'average_importance_cv'
        ]].to_string())

    # Calc performance metrics
    mean_cv_train_accuracy = np.mean(fold_train_accuracies)
    std_cv_train_accuracy = np.std(fold_train_accuracies)
    
    mean_cv_val_accuracy = np.mean(fold_val_accuracies)
    std_cv_val_accuracy = np.std(fold_val_accuracies)
    
    print(f"\nRandom Forest Cross-Validation Performance")
    print(f"\nMean Train Accuracy: \
        {mean_cv_train_accuracy:.4f} \
        (+/- {std_cv_train_accuracy:.4f})"
        )
    print(f"\nIndividual Fold Train Accuracies: \
        {[round(acc, 4) for acc in fold_train_accuracies]}"
        )
    print(f"\nMean Validation Accuracy: \
        {mean_cv_val_accuracy:.4f} \
        (+/- {std_cv_val_accuracy:.4f})"
        )
    print(f"\nIndividual Fold Validation Accuracies: \
        {[round(acc, 4) for acc in fold_val_accuracies]}"
        )


if __name__ == "__main__":
    main()