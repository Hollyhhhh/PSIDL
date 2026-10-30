"""Shared command-line settings and branch-decision output for simulations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy


def initialize_run(sample_id: int) -> None:
    if sample_id < 1:
        raise ValueError("sample ID must be positive")
    mode = os.environ.get("PSIDL_SIM_MODE", "random")
    if mode not in {"external", "internal", "random"}:
        raise ValueError("PSIDL_SIM_MODE must be external, internal or random")
    codons = int(os.environ.get("PSIDL_SIM_CODONS", "1000"))
    if codons < 64:
        raise ValueError("at least 64 codons are required")
    seed = os.environ.get("PSIDL_SIM_SEED")
    if seed is not None:
        seed_value = int(seed)
        if not 0 <= seed_value <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")
        random.seed(seed_value)
        numpy.random.seed(seed_value)


def choose_mode() -> str:
    mode = os.environ.get("PSIDL_SIM_MODE", "random")
    return random.choice(["internal", "external"]) if mode == "random" else mode


def validate_fitness(path: Path) -> int:
    values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(values) not in {60, 61}:
        raise ValueError("a fitness profile must contain exactly 60 or 61 numeric lines")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fitness values must be finite")
    return len(values)


def fitness_path(directory: str, profile: int) -> str:
    path = Path(os.environ.get("PSIDL_FITNESS_FILE", str(Path(directory) / f"{profile}.fitness")))
    validate_fitness(path)
    return str(path)


def record_branch(decisions: list, node, is_foreground: bool) -> None:
    decisions.append({
        "name": node.name,
        "is_leaf": bool(node.is_leaf()),
        "is_foreground": bool(is_foreground),
        "descendant_leaves": sorted(leaf.name for leaf in node.iter_leaves()),
    })


def write_audit(
    output_dir: str,
    sample_id: int,
    scenario: str,
    mode: str,
    codons: int,
    tree,
    decisions: list,
    foreground_species,
    foreground_nodes,
    parameters: dict,
    fitness: str | None = None,
) -> None:
    output = Path(output_dir)
    newick = output / "newick"
    files = {}
    patterns = {"tree": f"tree{sample_id}_*.rtree", "alignment": f"seq{sample_id}_*.newick", "sites": f"seq{sample_id}_*.sites", "branches": f"seq{sample_id}_*.branch"}
    for key, pattern in patterns.items():
        matches = list(newick.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one {key} file for sample {sample_id}, found {len(matches)}")
        files[key] = matches[0].name
    fitness_metadata = None
    if fitness is not None:
        path = Path(fitness)
        fitness_metadata = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "value_count": validate_fitness(path),
            "source": "explicit_file" if os.environ.get("PSIDL_FITNESS_FILE") else "fitness_library",
        }
    seed = os.environ.get("PSIDL_SIM_SEED")
    payload = {
        "schema_version": 1,
        "sample_id": int(sample_id),
        "scenario": scenario,
        "mode": mode,
        "requested_mode": os.environ.get("PSIDL_SIM_MODE", "random"),
        "seed": int(seed) if seed is not None else None,
        "codons": int(codons),
        "rate_blocks": int(os.environ.get("PSIDL_SIM_RATE_BLOCKS", "100")) if scenario in {"srv", "combined"} else None,
        "taxa": len(tree.get_leaves()),
        "requested_foreground_leaves": sorted(f"sequence{index}" for index in foreground_species),
        "requested_foreground_internal_nodes": [node.name for node in foreground_nodes],
        "branches": decisions,
        "outputs": files,
        "parameters": parameters,
        "fitness_input": fitness_metadata,
    }
    path = newick / f"sample_{sample_id}.audit.json"
    if path.exists():
        raise FileExistsError(f"sample audit already exists: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
