"""Predict positive selection for foreground branches in a codon alignment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import re
from pathlib import Path
import tomllib

import numpy as np
import torch

from psidl.alphabets import AminoAcid, Codon
from psidl.checkpoint import extract_model_state, torch_load
from psidl.config import ModelConfig
from psidl.genetics import AMINO_INDEX, CODON_INDEX, CODON_TO_AMINO
from psidl.model import SelectionModelForBranchClassification
from psidl.runtime import configure_runtime, resolve_device


_LEGACY_CODONS = """
TTT TCT TAT TGT TTC TCC TAC TGC TTA TCA TTG TCG TGG
CTT CCT CAT CGT CTC CCC CAC CGC CTA CCA CAA CGA CTG CCG CAG CGG
ATT ACT AAT AGT ATC ACC AAC AGC ATA ACA AAA AGA ATG ACG AAG AGG
GTT GCT GAT GGT GTC GCC GAC GGC GTA GCA GAA GGA GTG GCG GAG GGG
""".split()


def encoding_maps(encoding: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return raw codon and amino-acid indices for a recorded input convention."""
    if encoding == "empirical":
        return ({codon: i for i, codon in enumerate(sorted(CODON_TO_AMINO))},
                {amino: i for i, amino in enumerate(sorted(set(CODON_TO_AMINO.values())))})
    if encoding == "legacy-simulation":
        codons = {codon: i for i, codon in enumerate(_LEGACY_CODONS)}
        codons["GGA"] = 26  # Preserve the historical CCG/GGA token collision.
        return codons, {amino: i for i, amino in enumerate("FSYCLXWPHQRITNKMVADGE")}
    if encoding == "simulation":
        return dict(CODON_INDEX), dict(AMINO_INDEX)
    raise ValueError(f"unknown input encoding: {encoding}")


def read_fasta(path: Path) -> dict[str, str]:
    """Read aligned coding sequences, using the first FASTA header field as ID."""
    sequences: dict[str, str] = {}
    current = None
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            fields = line[1:].split()
            if not fields:
                raise ValueError(f"empty FASTA identifier on line {number}")
            current = fields[0]
            if current in sequences:
                raise ValueError(f"duplicate FASTA identifier: {current}")
            sequences[current] = ""
        else:
            if current is None:
                raise ValueError(f"sequence before first FASTA header on line {number}")
            sequences[current] += "".join(line.split()).upper().replace("U", "T")
    if not sequences or any(not sequence for sequence in sequences.values()):
        raise ValueError("FASTA contains no sequences or an empty sequence")
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError("FASTA sequences must have equal lengths")
    if next(iter(lengths)) % 3:
        raise ValueError("alignment length must be a multiple of three")
    if len(sequences) < 4:
        raise ValueError("at least four taxa are required")
    return sequences


def read_tree(path: Path):
    """Read named branches and optional HyPhy-style {test} markers."""
    try:
        from ete3 import Tree
    except ImportError as exc:
        raise ImportError('tree parsing requires: pip install -e ".[inference]"') from exc
    text = path.read_text(encoding="utf-8-sig").strip()
    text = re.sub(r"^\s*\[&R\]\s*", "", text, count=1, flags=re.IGNORECASE)
    if not text:
        raise ValueError("tree file is empty")
    try:
        tree = Tree(text, format=1, quoted_node_names=True)
    except Exception as exc:
        raise ValueError(f"invalid Newick tree: {exc}") from exc
    named, tagged = {}, []
    for index, node in enumerate(tree.traverse(strategy="preorder")):
        if "{test}" in node.name:
            if node.name.count("{test}") != 1 or not node.name.endswith("{test}"):
                raise ValueError("{test} must occur once at the end of a node name")
            node.name = node.name[:-6]
            tagged.append(node)
        if node.is_leaf() and not node.name:
            raise ValueError("tree contains an unnamed leaf")
        if node.name:
            if node.name in named:
                raise ValueError(f"duplicate or ambiguous tree node name: {node.name}")
            named[node.name] = node
        node.add_feature("prediction_id", node.name or f"internal:{index}")
    return tree, named, tagged


def prepare_inputs(
    fasta: Path,
    tree_path: Path,
    foreground: list[str] | None = None,
    encoding: str = "empirical",
) -> tuple[dict, dict]:
    """Resolve foreground descendants and remove globally invalid codon columns."""
    sequences = read_fasta(fasta)
    tree, named, tagged = read_tree(tree_path)
    leaves = {leaf.name for leaf in tree.iter_leaves()}
    if leaves != set(sequences):
        raise ValueError("FASTA identifiers and tree leaf names must match exactly")
    if foreground is None:
        selected = tagged
    else:
        if len(foreground) != len(set(foreground)):
            raise ValueError("foreground branch names must be unique")
        unknown = set(foreground) - named.keys()
        if unknown:
            raise ValueError(f"unknown foreground branch: {', '.join(sorted(unknown))}")
        selected = [named[name] for name in foreground]
    if not selected:
        raise ValueError("select --foreground branches or mark tree nodes with {test}")
    if any(node.is_root() for node in selected):
        raise ValueError("the root cannot be a foreground branch")
    foreground_taxa = {leaf.name for node in selected for leaf in node.iter_leaves()}
    if len(leaves - foreground_taxa) < 2:
        raise ValueError("at least two background taxa are required")
    names = list(sequences)
    order = ([name for name in names if name in foreground_taxa] +
             [name for name in names if name not in foreground_taxa])
    input_sites = len(sequences[names[0]]) // 3
    positions = [site for site in range(input_sites) if all(
        sequences[name][3 * site:3 * site + 3] in CODON_TO_AMINO for name in names)]
    if not positions:
        raise ValueError("alignment has no complete sense-codon columns after filtering")
    codon_index, amino_index = encoding_maps(encoding)
    triplets = [[sequences[name][3 * site:3 * site + 3] for site in positions] for name in order]
    codons = np.asarray([[codon_index[codon] + Codon.OFFSET for codon in row]
                         for row in triplets], dtype=np.int64)
    amino = np.asarray([[amino_index[CODON_TO_AMINO[codon]] + AminoAcid.OFFSET for codon in row]
                        for row in triplets], dtype=np.int64)
    markers = np.asarray([name in foreground_taxa for name in order], dtype=np.int64)
    inputs = {"codons": torch.from_numpy(codons[None]),
              "amino_acids": torch.from_numpy(amino[None]),
              "foreground_markers": torch.from_numpy(markers[None]), "nucleotides": None}
    metadata = {"selected_branches": [node.prediction_id for node in selected],
                "foreground_taxa": [name for name in order if name in foreground_taxa],
                "taxon_order": order, "taxon_count": len(order),
                "input_codon_count": input_sites, "kept_codon_count": len(positions),
                "removed_codon_count": input_sites - len(positions),
                "kept_codon_positions": [site + 1 for site in positions], "encoding": encoding}
    return inputs, metadata


def _model_config(checkpoint: dict, path: Path | None) -> ModelConfig:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    if path is not None:
        with path.open("rb") as handle:
            values = tomllib.load(handle).get("model")
        if not isinstance(values, dict):
            raise ValueError("model configuration must contain a [model] section")
    else:
        values = checkpoint.get("model_config")
        if values is None:
            native_config = checkpoint.get("config", {})
            values = native_config.get("model", {}) if isinstance(native_config, dict) else {}
        if isinstance(values, ModelConfig):
            values = asdict(values)
        if not isinstance(values, dict):
            raise ValueError("checkpoint model_config must be a dictionary")
    return ModelConfig(**values)


def predict(
    *, fasta: Path, tree: Path, checkpoint: Path, output: Path,
    foreground: list[str] | None = None, encoding: str = "empirical",
    device: str = "cpu", model_config: Path | None = None,
) -> dict:
    """Load a checkpoint and write one prediction for the foreground hypothesis."""
    paths = [fasta, tree, checkpoint] + ([model_config] if model_config is not None else [])
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("prediction output must differ from all input files")
    if output.exists():
        raise FileExistsError(f"prediction output already exists: {output}")
    inputs, result = prepare_inputs(fasta, tree, foreground, encoding)
    configure_runtime(seed=42, deterministic=True, cpu_threads=1)
    target = resolve_device(device)
    payload = torch_load(checkpoint, map_location="cpu")
    spec = _model_config(payload, model_config)
    if spec.use_nucleotide_embedding:
        raise ValueError("prediction supports checkpoints without nucleotide embeddings")
    model = SelectionModelForBranchClassification(spec)
    model.load_state_dict(extract_model_state(payload), strict=True)
    model.float().eval().to(target)
    inputs = {name: tensor.to(target) if tensor is not None else None for name, tensor in inputs.items()}
    with torch.inference_mode():
        logit = model(**inputs, dropout_ratio=0.0, return_dict=True).logits
        probability = logit.sigmoid()
    if not torch.isfinite(logit).all():
        raise RuntimeError("model produced a nonfinite prediction")
    result.update({"logit": float(logit.item()), "probability": float(probability.item()),
                   "device": str(target)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="codon-aligned coding sequences")
    parser.add_argument("--tree", required=True, type=Path, help="Newick tree with matching leaf names")
    parser.add_argument("--foreground", nargs="+", help="leaf or named internal branches; otherwise use {test}")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new JSON output file")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--encoding", choices=("empirical", "legacy-simulation", "simulation"), default="empirical")
    parser.add_argument("--model-config", type=Path, help="TOML file containing a [model] section")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = predict(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, ImportError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
