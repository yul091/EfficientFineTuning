
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


def get_nn_layers(module: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module]]:
    # DFS search
    layers = []
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Module) and list(child.children()):
            layers.extend(get_nn_layers(child, child_prefix))
        else:
            layers.append((child_prefix, child))
    return layers