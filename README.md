# PSIDL

Positive Selection Inference by Deep Learning

PSIDL predicts support for positive selection in a specified foreground branch
or branch set, using a codon-aligned FASTA file and a Newick tree. NONE and CUB
model weights are included in `models/`.

## Installation

Python 3.12 is required. Download and extract this repository using
**Code → Download ZIP**, then open a terminal in the folder containing
`pyproject.toml`. Alternatively, clone it with Git:

```bash
git clone https://github.com/Hollyhhhh/PSIDL.git
cd PSIDL
```

Create an environment and install PSIDL. The `python` command on Windows must
refer to Python 3.12.

**Windows — Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[inference]"
```

**Linux / macOS — bash:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[inference]"
```

Keep the environment active and run the commands below from the repository root.

## Input files

- **FASTA:** codon-aligned coding DNA sequences of equal length, divisible by
  three. The first word after `>` is the taxon ID.
- **Tree:** a rooted Newick tree whose leaf IDs match the FASTA IDs exactly.
- **Foreground:** the name of the terminal or internal branch to score.

Use at least four taxa, including at least two background taxa. Codon columns
containing a gap, ambiguous base or stop codon in any taxon are removed from all
sequences. At least one complete sense-codon column must remain.

The example files are `examples/example.fasta` and `examples/example.nwk`:

```text
((taxon_A:0.1,taxon_B:0.1)clade_AB:0.1,(taxon_C:0.1,taxon_D:0.1)clade_CD:0.1)root;
```

`taxon_A` selects its terminal branch. `clade_AB` selects the internal branch
leading to taxa A and B; both descendant taxa receive the foreground marker.
The root cannot be selected. Tree branch lengths are not neural-network inputs.

## Run PSIDL

Run PSIDL-NONE on the supplied example:

```bash
python -m psidl.predict --fasta examples/example.fasta --tree examples/example.nwk --foreground taxon_A --checkpoint models/none.pt --output results/none.json
```

Use PSIDL-CUB on the same foreground:

```bash
python -m psidl.predict --fasta examples/example.fasta --tree examples/example.nwk --foreground taxon_A --checkpoint models/cub.pt --output results/cub.json
```

Score an internal foreground branch:

```bash
python -m psidl.predict --fasta examples/example.fasta --tree examples/example.nwk --foreground clade_AB --checkpoint models/cub.pt --output results/internal.json
```

For your own data, replace the FASTA path, tree path and foreground name.
Multiple branch names may follow `--foreground`. If the tree already marks the
intended nodes with `{test}`, omit `--foreground`; an explicit list overrides
those marks. Use the default input encoding with the bundled models.

Prediction uses the CPU by default. To use an available CUDA GPU, append
`--device cuda`. Run `python -m psidl.predict --help` for command options.

## Read the result

The output directory is created automatically. Open the JSON file specified by
`--output`:

| Field | Meaning |
|---|---|
| `probability` | Model support for positive selection, between 0 and 1; higher values indicate stronger support. This is not a p-value. |
| `logit` | Score before the sigmoid transformation. |
| `foreground_taxa` | Taxa marked as foreground for this prediction. |
| `taxon_order` | Sequence order passed to the model. |
| `kept_codon_positions` | Retained codon positions in the original alignment, numbered from 1. |
| `removed_codon_count` | Number of codon columns removed from the input. |

Existing output files are preserved. Use a new `--output` filename when running
another prediction. All remaining sites and taxa are used without random
subsampling or cropping.

Citation details are in `CITATION.cff`. License information is in
[LICENSE_STATUS.md](LICENSE_STATUS.md).
