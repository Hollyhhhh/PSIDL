"""PSIDL model architecture.

The module nesting and parameter names intentionally match the legacy
``SelectionModelForBranchClassification`` checkpoint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, Module, ModuleList, Parameter

from psidl.alphabets import AminoAcid, Codon, Nucleotide
from psidl.config import ModelConfig

EPSILON = float(torch.finfo(torch.float32).eps)


def _activation(name: str, x: Tensor) -> Tensor:
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    if name == "silu":
        return F.silu(x)
    raise ValueError(f"unsupported activation: {name}")


@dataclasses.dataclass
class SelectionModelOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None

    def to_tuple(self) -> tuple[Tensor | tuple[Tensor, ...] | None, ...]:
        return tuple(value for value in (self.last_hidden_state, self.hidden_states) if value is not None)


@dataclasses.dataclass
class SelectionModelOutputForBranchClassification:
    logits: Tensor
    loss: Tensor | None = None
    last_hidden_state: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None

    def to_tuple(self) -> tuple[Tensor | tuple[Tensor, ...] | None, ...]:
        return tuple(
            value
            for value in (self.logits, self.loss, self.last_hidden_state, self.hidden_states)
            if value is not None
        )


class BaseModel(Module):
    @staticmethod
    def freeze(model: Module) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    def initialize_weights(self, model: Module, spec: ModelConfig) -> None:
        for module in model.modules():
            if isinstance(module, Embedding):
                torch.nn.init.trunc_normal_(module.weight)
                if module.padding_idx is not None:
                    torch.nn.init.zeros_(module.weight.data[module.padding_idx])
            elif isinstance(module, RMSNorm):
                torch.nn.init.normal_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, Linear):
                torch.nn.init.trunc_normal_(module.weight, std=spec.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


class RMSNorm(Module):
    def __init__(self, feature_count: int, eps: float = EPSILON):
        super().__init__()
        self.epsilon = eps
        self.feature_count = feature_count
        self.scale = float(feature_count) ** -0.5
        self.weight = Parameter(torch.ones(feature_count))
        self.bias = Parameter(torch.zeros(feature_count))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * self.scale + self.epsilon
        return x / rms * self.weight + self.bias


class MultiheadSelfAttention(Module):
    def __init__(self, head_count: int, hidden_size: int):
        super().__init__()
        if hidden_size % head_count:
            raise ValueError("hidden_size must be divisible by head_count")
        self.hidden_size = hidden_size
        self.head_count = head_count
        self.head_dim = hidden_size // head_count
        self.qkv_proj = Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, dropout_ratio: float = 0.0) -> Tensor:
        # The legacy implementation explicitly disabled attention-probability dropout.
        attention_dropout = 0.0
        qkv = self.qkv_proj(x).unflatten(-1, (3, self.head_count, self.head_dim))
        qkv = qkv.transpose(1, 3)
        output = F.scaled_dot_product_attention(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            dropout_p=attention_dropout,
        )
        output = output.transpose(1, 2).flatten(2, 3)
        return self.out_proj(output)


class MultidimensionalAttentionBlock(Module):
    def __init__(
        self,
        head_count: int,
        hidden_size: int,
        activation: str,
        feed_forward_hidden_size: int,
        dimension: int = -2,
    ):
        super().__init__()
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.head_count = head_count
        self.head_dim = hidden_size // head_count
        self.norm1 = RMSNorm(hidden_size)
        self.attention = MultiheadSelfAttention(head_count, hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.linear1 = Linear(hidden_size, feed_forward_hidden_size)
        self.activation_name = activation
        self.linear2 = Linear(feed_forward_hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor, dropout_ratio: float = 0.0) -> Tensor:
        x = hidden_states
        dimension_count = x.ndim
        if self.dimension != -2 and self.dimension != dimension_count - 2:
            x = x.transpose(self.dimension, -2)
        shape = x.size()
        x = x.flatten(0, -3)

        output = self.norm1(x)
        output = self.attention(output, dropout_ratio=dropout_ratio)
        output = F.dropout(output, dropout_ratio, training=self.training, inplace=True)
        x = output + x

        output = self.norm2(x)
        output = self.linear1(output)
        output = _activation(self.activation_name, output)
        output = self.linear2(output)
        output = F.dropout(output, dropout_ratio, training=self.training, inplace=True)
        x = output + x

        x = x.view(*shape)
        if self.dimension != -2 and self.dimension != dimension_count - 2:
            x = x.transpose(self.dimension, -2)
        return x


class SelectionEmbedder(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.foreground_marker_embedder = Embedding(2, config.hidden_size)
        self.amino_acid_embedder = Embedding(AminoAcid.ALPHABET_SIZE, config.hidden_size)
        self.codon_embedder = Embedding(Codon.ALPHABET_SIZE, config.hidden_size)
        self.nucleotide_embedder = (
            Embedding(Nucleotide.ALPHABET_SIZE, config.hidden_size)
            if config.use_nucleotide_embedding
            else None
        )

    def forward(
        self,
        foreground_markers: Tensor,
        amino_acids: Tensor,
        codons: Tensor,
        nucleotides: Tensor | None,
    ) -> Tensor:
        x = self.foreground_marker_embedder(foreground_markers[..., None])
        x = x + self.amino_acid_embedder(amino_acids)
        x = x + self.codon_embedder(codons)
        if self.nucleotide_embedder is not None and nucleotides is not None:
            x = x + self.nucleotide_embedder(nucleotides)
        return x


class SelectionTransformerBackbone(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size)
        self.blocks = ModuleList(
            [
                MultidimensionalAttentionBlock(
                    config.num_attention_heads,
                    config.hidden_size,
                    config.feed_forward_activation,
                    config.feed_forward_hidden_size,
                    # This field existed in the original configuration but was
                    # unused: the checkpoint architecture always starts with taxa.
                    dimension=layer % 2 - 3,
                )
                for layer in range(config.num_attention_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        dropout_ratio: float = 0.0,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        layer_outputs: list[Tensor] = []
        x = hidden_states
        for block in self.blocks:
            x = block(x, dropout_ratio=dropout_ratio)
            if output_hidden_states:
                layer_outputs.append(x)
        x = self.norm(x)
        return x, tuple(layer_outputs) if output_hidden_states else None


class SelectionModel(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedder = SelectionEmbedder(config)
        self.backbone = SelectionTransformerBackbone(config)

    def forward(
        self,
        foreground_markers: Tensor,
        amino_acids: Tensor,
        codons: Tensor,
        nucleotides: Tensor | None,
        dropout_ratio: float = 0.0,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SelectionModelOutput | Sequence[Tensor | tuple[Tensor, ...] | None]:
        embeddings = self.embedder(
            foreground_markers=foreground_markers,
            amino_acids=amino_acids,
            codons=codons,
            nucleotides=nucleotides,
        )
        features, attention_outputs = self.backbone(
            embeddings,
            dropout_ratio=dropout_ratio,
            output_hidden_states=output_hidden_states,
        )
        result = SelectionModelOutput(features, attention_outputs)
        return result if return_dict else result.to_tuple()


class SelectionBranchPredictionHead(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = Linear(config.hidden_size, config.feed_forward_hidden_size)
        self.activation_name = config.feed_forward_activation
        self.linear2 = Linear(config.feed_forward_hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        x = self.linear1(features)
        x = _activation(self.activation_name, x)
        x = self.linear2(x)[..., 0]
        return x.mean(dim=(1, 2))


class SelectionModelForBranchClassification(BaseModel):
    def __init__(self, spec: ModelConfig):
        super().__init__()
        self.spec = spec
        self.backbone = SelectionModel(spec)
        self.classifier = SelectionBranchPredictionHead(spec)
        self.initialize_weights(self.backbone, self.spec)
        self.initialize_weights(self.classifier, self.spec)

    def forward(
        self,
        foreground_markers: Tensor,
        amino_acids: Tensor,
        codons: Tensor,
        nucleotides: Tensor | None,
        labels: Tensor | None = None,
        dropout_ratio: float = 0.0,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SelectionModelOutputForBranchClassification | Sequence:
        features = self.backbone(
            foreground_markers=foreground_markers,
            amino_acids=amino_acids,
            codons=codons,
            nucleotides=nucleotides,
            dropout_ratio=dropout_ratio,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if not isinstance(features, SelectionModelOutput):
            raise TypeError("unexpected backbone output")
        logits = self.classifier(features.last_hidden_state)
        loss = (
            None
            if labels is None
            else F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        )
        result = SelectionModelOutputForBranchClassification(
            logits=logits,
            loss=loss,
            last_hidden_state=features.last_hidden_state,
            hidden_states=features.hidden_states,
        )
        return result if return_dict else result.to_tuple()


def count_parameters(module: Module) -> int:
    return sum(
        int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad
    )
