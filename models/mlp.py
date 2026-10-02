"""Compact MLP search models and exact parameter estimates."""
from torch import nn
from data.specs import INPUT_FEATURES, NUM_CLASSES


class HiddenLayer(nn.Module):
    def __init__(self, incoming, outgoing, arch):
        super().__init__()
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[arch["activation"]]
        self.layers = nn.Sequential(nn.Linear(incoming, outgoing),
                                    nn.LayerNorm(outgoing) if arch["layer_norm"] else nn.Identity(),
                                    activation(), nn.Dropout(arch["dropout"]))
        self.residual = arch["use_residual"] and incoming == outgoing

    def forward(self, x):
        output = self.layers(x)
        return output + x if self.residual else output


def build_model(arch, input_features=INPUT_FEATURES, num_classes=NUM_CLASSES):
    layers = []
    incoming = input_features
    for i in range(1, arch["num_layers"] + 1):
        outgoing = arch[f"width_{i}"]
        layers.append(HiddenLayer(incoming, outgoing, arch))
        incoming = outgoing
    return nn.Sequential(*layers, nn.Linear(incoming, num_classes))


def estimate_parameters(arch, input_features=INPUT_FEATURES, num_classes=NUM_CLASSES):
    total, incoming = 0, input_features
    for i in range(1, arch["num_layers"] + 1):
        outgoing = arch[f"width_{i}"]
        total += incoming * outgoing + outgoing + (2 * outgoing if arch["layer_norm"] else 0)
        incoming = outgoing
    return total + incoming * num_classes + num_classes


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_baseline_mlp(input_features=INPUT_FEATURES, num_classes=NUM_CLASSES):
    return build_model({"num_layers": 2, "width_1": 64, "width_2": 32,
                        "activation": "relu", "dropout": 0.1,
                        "layer_norm": False, "use_residual": False}, input_features, num_classes)
