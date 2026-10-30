"""Run codon simulations with explicit modes, seeds and output paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from helper import validate_fitness

SCRIPTS = {
    "none": "sim_none.py",
    "cub": "sim_cub.py",
    "dns": "sim_dns.py",
    "srv": "sim_srv.py",
    "combined": "sim_3fac.py",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def seed_int(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCRIPTS, required=True)
    parser.add_argument("--sample-id", type=positive_int, default=1,
                        help="First sample identifier (default: 1).")
    parser.add_argument("--count", type=positive_int, default=1,
                        help="Number of sequential samples (default: 1).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("external", "internal", "random"), default="random")
    parser.add_argument("--codons", type=positive_int, default=1000)
    parser.add_argument("--rate-blocks", type=positive_int, default=100,
                        help="SRV/combined rate blocks; use a small count for execution examples.")
    parser.add_argument("--seed", type=seed_int, help="First sample seed; incremented for each sample.")
    fitness_group = parser.add_mutually_exclusive_group()
    fitness_group.add_argument("--fitness-dir", type=Path)
    fitness_group.add_argument("--fitness-file", type=Path,
                               help="Use one supplied 60/61-value profile instead of a library.")
    args = parser.parse_args()
    if args.codons < 64:
        parser.error("--codons must be at least 64")
    if args.seed is not None and args.seed + args.count - 1 > 2**32 - 1:
        parser.error("per-sample seed range would exceed 4294967295")
    if args.scenario not in {"srv", "combined"} and args.rate_blocks != 100:
        parser.error("--rate-blocks applies to srv or combined")
    if args.scenario in {"srv", "combined"} and (
        args.codons < args.rate_blocks or args.codons % args.rate_blocks
    ):
        parser.error("srv/combined --codons must be a positive multiple of --rate-blocks")

    output = args.output_dir.expanduser().resolve()
    environment = os.environ.copy()
    environment["PSIDL_SIM_OUTPUT"] = str(output)
    environment["PSIDL_SIM_MODE"] = args.mode
    environment["PSIDL_SIM_CODONS"] = str(args.codons)
    environment["PSIDL_SIM_RATE_BLOCKS"] = str(args.rate_blocks)
    environment.pop("PSIDL_SIM_SEED", None)
    environment.pop("PSIDL_FITNESS_FILE", None)
    for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if args.scenario in {"cub", "combined"}:
        if args.fitness_file is not None:
            fitness = args.fitness_file.expanduser().resolve()
            try:
                validate_fitness(fitness)
            except (OSError, ValueError) as error:
                parser.error(str(error))
            environment["PSIDL_FITNESS_FILE"] = str(fitness)
        else:
            if args.fitness_dir is None:
                parser.error("--fitness-dir or --fitness-file is required for cub and combined")
            fitness = args.fitness_dir.expanduser().resolve()
            if not fitness.is_dir():
                parser.error(f"fitness directory does not exist: {fitness}")
            environment["PSIDL_FITNESS_DIR"] = str(fitness)
    elif args.fitness_file is not None or args.fitness_dir is not None:
        parser.error("fitness inputs apply to cub or combined")

    identifiers = range(args.sample_id, args.sample_id + args.count)
    for identifier in identifiers:
        existing = any((output / "newick").glob(f"seq{identifier}_*"))
        existing = existing or any((output / "newick").glob(f"tree{identifier}_*"))
        existing = existing or (output / "newick" / f"sample_{identifier}.audit.json").exists()
        if existing:
            parser.error(f"sample {identifier} already has outputs in {output / 'newick'}")

    script = Path(__file__).resolve().parent / SCRIPTS[args.scenario]
    for identifier in identifiers:
        if args.seed is not None:
            environment["PSIDL_SIM_SEED"] = str(args.seed + identifier - args.sample_id)
        print(f"Running {args.scenario} sample {identifier}", flush=True)
        subprocess.run([sys.executable, str(script), str(identifier)],
                       env=environment, check=True)
    print(f"Saved {args.count} sample(s) under {output / 'newick'}")


if __name__ == "__main__":
    main()
