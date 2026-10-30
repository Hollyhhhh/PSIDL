"""Token definitions used by the released PSIDL checkpoints."""

NUCLEOTIDES = "ACGT"
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"


class Nucleotide:
    ALPHABET_SIZE = 16
    PADDING = 0
    MASK = 1
    OFFSET = 2
    GAP = 6
    END = 7
    LABEL_OFFSET = 8


class AminoAcid:
    ALPHABET_SIZE = 32
    PADDING = 0
    MASK = 1
    OFFSET = 2
    GAP = 22
    END = 23
    LABEL_OFFSET = 24


class Codon:
    ALPHABET_SIZE = 73
    PADDING = 0
    MASK = 2
    OFFSET = 3
    GAP = 64
    END = 65
    LABEL_OFFSET = 66
