import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, List


class FullyConnectedLayer(nn.Module):
    """A flexible fully connected layer with various customisation options.

    Args:
        in_features (int): Num input features.
        out_features (int): Num output features.
        bias (bool, optional): Include bias term? Defaults to True.
        activation (str, optional): Activation function to use. Defaults to 
            'linear'.
        weight_init (str): Weight initialisation method. Defaults to 'xavier'.
        dropout (float): Dropout rate. Defaults to 0.0.
        batch_norm (bool, optional): Use batch norm? Default to False.
        layer_norm (bool, optional): Use layer norm? Default to False.

    REFERENCES:
    (0) I previously created this class for a project submitted for 
        COMP3710 at UQ.
    (1) This code was developed with assistance from the Claude AI assistant,
        created by Anthropic, PBC. Claude provided guidance on implementing
        StyleGAN2 architecture and training procedures.
        Date of assistance: 8-21/10/2024
        Claude version: Claude-3.5 Sonnet
        For more information about Claude: https://www.anthropic.com
    (2) GitHub Repository: stylegan2-ada-pytorch
        URL: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main
        Accessed on: 29/09/24 - 8/10/24
    (3) Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & 
        Aila, T. (2020). 
        Analyzing and improving the image quality of StyleGAN.
        arXiv. https://arxiv.org/abs/1912.04958
    (4) Karras, T., Laine, S., & Aila, T. (2019).
        A Style-Based Generator Architecture for Generative Adversarial 
        Networks. arXiv. https://arxiv.org/abs/1812.04948
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True, 
            activation: str = 'linear', 
            weight_init: str = 'xavier',
            dropout: float = 0.0, 
            batch_norm: bool = False,
            layer_norm: bool = False
            ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None

        # Init weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.bias = None

        self.reset_parameters(weight_init)
        self.act_fn = self.get_activation_fn(activation)

    def reset_parameters(self, weight_init: str) -> None:
        """Initialise the weights using the specified method."""
        if weight_init.lower() == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        elif weight_init.lower() == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        elif weight_init.lower() == 'orthogonal':
            nn.init.orthogonal_(self.weight)
        else:
            raise ValueError(f"Unsupported weight init: {weight_init}")

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_activation_fn(
            self,
            activation: str
            ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the specified activation function."""
        if activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'leaky_relu':
            return lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif activation.lower() == 'elu':
            return F.elu
        elif activation.lower() == 'gelu':
            return F.gelu
        elif activation.lower() == 'silu':
            return F.silu
        elif activation.lower() == 'tanh':
            return torch.tanh
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'linear':
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fully connected layer."""
        x = F.linear(x, self.weight, self.bias)

        if self.batch_norm:
            x = self.batch_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)

        x = self.act_fn(x)

        if self.dropout:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    """Flexible multi-layer perceotron."""
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_units: List[int],
                 activation_fn: str = 'linear',
                 weight_init: str = 'xavier',
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 is_binary: bool = False
                 ) -> None:
        """
        Args:
            input_dim (int): Dimension of input features.
            num_classes (int): Number of output classes.
            hidden_units (list[int]): Neurons in each hidden layer.
            activation (str): Activation for hidden layers.
            weight_init (str): Weight init for all layers.
            dropout_rate (float): Dropout rate for hidden layers.
            use_batch_norm (bool): Apply batch norm to hidden layers.
            use_layer_norm (bool): Apply layer norm to hidden layers.
            is_binary (bool): If True, output dim = 1 (for BCEWithLogitsLoss).
        """
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim

        #Hidden layers
        for h_units in hidden_units:
            self.layers.append(
                FullyConnectedLayer(
                    in_features=current_dim,
                    out_features=h_units,
                    activation=activation_fn,
                    weight_init=weight_init,
                    dropout=dropout_rate,
                    batch_norm=use_batch_norm,
                    layer_norm=use_layer_norm,
                    )
                )
            current_dim = h_units # Update for next layer
        
        # Output
        output_dim = 1 if is_binary else num_classes
        self.layers.append(
            FullyConnectedLayer(
                in_features=current_dim,
                out_features=output_dim,
                activation='linear',
                weight_init=weight_init,
                dropout=0.0,
                batch_norm=False,
                layer_norm=False
                )
                ) # Unholy I know. Still follows PEP 8 tho.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

