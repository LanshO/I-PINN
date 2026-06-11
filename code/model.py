from collections import OrderedDict

import torch
import torch.nn as nn


ACTIVATIONS = {
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
}


class ResidualBlock(nn.Module):
    """Residual block used by the single-hole I-PINN backbone."""

    def __init__(self, hidden_size, act):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = act()

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.act(out)
        out += residual
        return out


class ResNN(nn.Module):
    """Residual MLP used by the validated single-hole inverse workflow."""

    def __init__(self, input_size, hidden_size, output_size, depth, act):
        super().__init__()
        layers = [("input", nn.Linear(input_size, hidden_size))]
        layers.append(("input_activation", act()))
        for i in range(depth):
            layers.append((f"residual_{i}", ResidualBlock(hidden_size, act)))
        layers.append(("output", nn.Linear(hidden_size, output_size)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)


class PlainNN(nn.Module):
    """Plain MLP kept for parity with the original research script."""

    def __init__(self, input_size, hidden_size, output_size, depth, act=torch.nn.Tanh):
        super().__init__()
        layers = [("input", torch.nn.Linear(input_size, hidden_size))]
        layers.append(("input_activation", act()))
        for i in range(depth):
            layers.append((f"hidden_{i}", torch.nn.Linear(hidden_size, hidden_size)))
            layers.append((f"activation_{i}", act()))
        layers.append(("output", torch.nn.Linear(hidden_size, output_size)))
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)


def initialize_model(model):
    """Initialize the network exactly as in the validated single-hole script."""

    def gelu_init(tensor):
        nn.init.normal_(
            tensor,
            mean=0.0,
            std=torch.sqrt(torch.tensor(2.0 / tensor.size(1), dtype=torch.float32)),
        )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "residual" in name:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif "output" in name:
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            else:
                gelu_init(module.weight)
            nn.init.zeros_(module.bias)


def build_model(name="resnn", activation="gelu", input_size=2, hidden_size=64, output_size=2, depth=8):
    """Build the network backbone requested by the framework entry script."""

    activation_name = activation.lower()
    if activation_name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    act = ACTIVATIONS[activation_name]

    model_name = name.lower()
    if model_name == "resnn":
        model = ResNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, depth=depth, act=act)
    elif model_name == "plainnn":
        model = PlainNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, depth=depth, act=act)
    else:
        raise ValueError(f"Unsupported model: {name}")

    initialize_model(model)
    return model
