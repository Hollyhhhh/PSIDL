"""Model settings for PSIDL prediction."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    initializer_range: float = 0.02
    hidden_size: int = 384
    use_nucleotide_embedding: bool = False
    sequence_attention_first: bool = False
    num_attention_layers: int = 6
    num_attention_heads: int = 6
    feed_forward_activation: str = "gelu_pytorch_tanh"
    feed_forward_hidden_size: int = 768

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.feed_forward_activation not in {"gelu", "gelu_pytorch_tanh", "relu", "silu"}:
            raise ValueError(f"unsupported activation: {self.feed_forward_activation}")
