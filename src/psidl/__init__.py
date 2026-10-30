"""PSIDL: positive-selection inference with multidimensional attention."""

from typing import Any

__all__ = ["ModelConfig", "SelectionModelForBranchClassification"]
__version__ = "0.3.0"


def __getattr__(name: str) -> Any:
    if name == "ModelConfig":
        from psidl.config import ModelConfig

        return ModelConfig
    if name == "SelectionModelForBranchClassification":
        from psidl.model import SelectionModelForBranchClassification

        return SelectionModelForBranchClassification
    raise AttributeError(name)
