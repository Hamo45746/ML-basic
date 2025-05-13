import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
from data_loader import stratified_split, select_features, load_data
from model_builder import MLP
from preprocessing import normalise_features, create_dataloaders
from utils import set_seeds, plot_confusion_mat, create_tensorboard_writer
from training import select_device, define_optimiser, \
    run_training_loop, test_set_eval


DATA_PATH = "data/TTSwing.csv"
RESULTS_DIR = "results"
TENSORBOARD_LOG_DIR = os.path.join(RESULTS_DIR, "tensorboard_logs")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
TARGET_COL = 'testmode'
RANDOM_SEED = 1
FEATURE_IMPORTANCES_DICT = {
    'a_min': 0.153130, 'g_entropy': 0.096646, 'a_entropy': 0.076896,
    'g_mean': 0.070897, 'g_skewn': 0.064538, 'gz_var': 0.060214,
    'a_mean': 0.056541, 'gz_mean': 0.045354, 'a_skewn': 0.030751,
    'gy_mean': 0.025196, 'gy_var': 0.022969, 'ax_mean': 0.022098,
    'ay_rms': 0.021941, 'gz_rms': 0.019492, 'a_max': 0.019424,
    'az_var': 0.018903, 'ax_rms': 0.018580, 'g_max': 0.017785,
    'g_fft': 0.017099, 'ay_mean': 0.014708, 'gx_mean': 0.013919,
    'ay_var': 0.013714, 'gy_rms': 0.011878, 'a_psdx': 0.010636,
    'az_rms': 0.010007, 'gx_rms': 0.009841, 'a_fft': 0.009476,
    'g_kurt': 0.009198, 'ax_var': 0.008837, 'gx_var': 0.008581,
    'a_kurt': 0.006331, 'g_psdx': 0.006284, 'az_mean': 0.005124,
    'g_min': 0.003013
    }
FEATURE_NAMES_ORDERED = list(FEATURE_IMPORTANCES_DICT.keys())


def run_single_experiment(
        config: Dict[str, Any],
        experiment_idx: int,
        device: torch.device
        ):
    """Run single experiment with given config."""
    run_id = config.get("id", f"exp_{experiment_idx}")
    print(f"\nExperiment: {run_id}")

    set_seeds(RANDOM_SEED)

    # Data Loading and Preprocessing
    df = load_data(DATA_PATH)
    num_classes = df[TARGET_COL].nunique()
    class_names = [str(c) for c in sorted(df[TARGET_COL].unique())]
    features = FEATURE_NAMES_ORDERED[:config['top_n_features']]
    
    X, y = select_features(df, features, TARGET_COL)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X,
        y,
        test_ratio=config.get('test_split_size', 0.2), 
        validation_ratio=config.get('val_split_size', 0.1),
        random_state=RANDOM_SEED
        )

    X_train_norm, X_val_norm, X_test_norm, scaler = normalise_features(
        X_train,
        X_val,
        X_test
        )

    # class weights for imbalance
    y_train_np = y_train.to_numpy()
    class_weights_tensor = None
    if config.get('use_class_weights', True):
        unique_classes = np.unique(y_train_np) #sorted list of classes
        weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train_np
                )
        class_weights_tensor = torch.tensor(
            weights,
            dtype=torch.float
            ).to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_norm,
        X_val_norm,
        X_test_norm,
        y_train.to_numpy(dtype=np.int64),
        y_val.to_numpy(dtype=np.int64),
        y_test.to_numpy(dtype=np.int64),
        batch_size=config['batch_size']
        )

    # Model Definition
    model = MLP(
        input_dim=X_train_norm.shape[1],
        num_classes=num_classes,
        hidden_units=config['hidden_units'],
        activation_fn=config['activation_fn'],
        weight_init=config.get('weight_init', 'xavier'),
        dropout_rate=config.get('dropout_rate', 0.0),
        use_batch_norm=config.get('use_batch_norm', False),
        use_layer_norm=config.get('use_layer_norm', False),
        is_binary=False
        ).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimiser = define_optimiser(
        model,
        optimiser_name=config['optimiser'],
        lr=config['lr']
        )
    
    writer_group = config.get("experiment_group_name", "Hyperparams_TestMode")
    writer = create_tensorboard_writer(
        TENSORBOARD_LOG_DIR,
        writer_group,
        run_id
        )

    #Training Loop
    history = run_training_loop(
        model,
        train_loader,
        val_loader,
        optimiser,
        loss_fn,
        device,
        epochs=config['epochs'],
        writer=writer
        )

    # Eval on Test Set
    if test_loader:
        test_loss, test_acc, test_preds, test_true_labels = test_set_eval(
            model,
            test_loader,
            loss_fn,
            device
            )
        print(f"Test Eval: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        writer.add_scalar('Loss/test', test_loss, config['epochs'])
        writer.add_scalar('Accuracy/test', test_acc, config['epochs'])

        cm = confusion_matrix(
            test_true_labels,
            test_preds,
            labels=np.arange(num_classes)
            )

        print("\nConfusion Matrix:")
        print(cm)
        cm_title = f'CM - {run_id}'
        cm_save_path = os.path.join(
            CONFUSION_MATRIX_DIR,
            writer_group,
            f"{run_id}_cm.png"
            )
        plot_confusion_mat(
            cm,
            class_names,
            title=cm_title,
            save_path=cm_save_path
            )
        
        # Add CM to TensorBoard
        fig_cm, ax_cm = plt.subplots()
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
            )
        cm_display.plot(ax=ax_cm, cmap=plt.cm.Blues)
        ax_cm.set_title(cm_title)
        writer.add_figure(
            f'ConfusionMatrix/test',
            fig_cm,
            global_step=config['epochs']
            )
        plt.close(fig_cm)
    else:
        print("No test loader. Skipping test set evaluation.")

    writer.close()


if __name__ == "__main__":
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)

    DEVICE = select_device()

    # Def experiment configs
    # Base config shared by all
    base_config = {
        "top_n_features": 9,
        "batch_size": 64,
        "epochs": 30,
        "optimiser": "adam",
        "test_split_size": 0.2,
        "val_split_size": 0.15,
        "use_class_weights": True,
        "experiment_group_name": "run2"
        }

    experiments_configs = [
        # Baseline
        {
            **base_config,
            "id": "baseline",
            "hidden_units": [16, 16],
            "activation_fn": "relu",
            "lr": 0.001
            },
        
        # Vary depth/breadth
        # {
        #     **base_config,
        #     "id": "deeper_narrower",
        #     "hidden_units": [8, 8, 8],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },
        # {
        #     **base_config,
        #     "id": "wider_shallower",
        #     "hidden_units": [32, 32],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },
        # {
        #     **base_config,
        #     "id": "single_wide_layer",
        #     "hidden_units": [64],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },

        # Vary activations
        {
            **base_config,
            "id": "activation_elu",
            "hidden_units": [16, 16],
            "activation_fn": "elu", "lr": 0.001
            },
        {
            **base_config,
            "id": "activation_tanh",
            "hidden_units": [16, 16],
            "activation_fn": "tanh",
            "lr": 0.001},
        {
            **base_config,
            "id": "activation_silu",
            "hidden_units": [16, 16],
            "activation_fn": "silu",
            "lr": 0.001
            },
        {
            **base_config,
            "id": "activation_leaky_relu",
            "hidden_units": [16, 16],
            "activation_fn": "leaky_relu",
            "lr": 0.001
            },
        {
            **base_config,
            "id": "activation_gelu",
            "hidden_units": [16, 16],
            "activation_fn": "gelu",
            "lr": 0.001
            },
        {
            **base_config,
            "id": "activation_sigmoid",
            "hidden_units": [16, 16],
            "activation_fn": "sigmoid",
            "lr": 0.001
            },
        {
            **base_config,
            "id": "activation_linear",
            "hidden_units": [16, 16],
            "activation_fn": "linear",
            "lr": 0.001
            },

        # Vary learning rate
        # {
        #     **base_config,
        #     "id": "lr_high",
        #     "hidden_units": [16, 16], 
        #     "activation_fn": "relu",
        #     "lr": 0.01
        #     },
        # {
        #     **base_config,
        #     "id": "lr_low",
        #     "hidden_units": [16, 16],
        #     "activation_fn": "relu",
        #     "lr": 0.0001
        #     },
        
        # Vary num features
        # {
        #     **base_config,
        #     "id": "features_top5",
        #     "top_n_features": 5,
        #     "hidden_units": [16, 16],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },
        # {
        #     **base_config,
        #     "id": "features_top15",
        #     "top_n_features": 15,
        #     "hidden_units": [16, 16],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },
        # {
        #     **base_config,
        #     "id": "features_all34",
        #     "top_n_features": 34,
        #     "hidden_units": [16, 16],
        #     "activation_fn": "relu",
        #     "lr": 0.001
        #     },
            ]

    for idx, config in enumerate(experiments_configs):
        run_single_experiment(config, idx, DEVICE)
