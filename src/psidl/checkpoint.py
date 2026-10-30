"""Load PSIDL model parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    elif "module" in checkpoint and isinstance(checkpoint["module"], dict):
        state = checkpoint["module"]
    elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state = checkpoint
    else:
        raise KeyError("checkpoint contains neither a 'model' nor a 'module' state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state
