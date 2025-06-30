import collections
import pandas as pd
import numpy as np
import os

# --- Constants for Genetic Code and Amino Acid Weights ---

STANDARD_GENETIC_CODE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Average molecular weights of amino acid residues (subtracting H2O)
# Source: https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/protein-biology/protein-purification/amino-acid-molecular-weights
AMINO_ACID_WEIGHTS = {
    'A': 71.0788, 'C': 103.1388, 'D': 115.0886, 'E': 129.1155,
    'F': 147.1766, 'G': 57.0519, 'H': 137.1411, 'I': 113.1594,
    'K': 128.1741, 'L': 113.1594, 'M': 131.1926, 'N': 114.1038,
    'P': 97.1167, 'Q': 128.1307, 'R': 156.1875, 'S': 87.0782,
    'T': 101.1051, 'V': 99.1326, 'W': 186.2132, 'Y': 163.1760
}

# Hydropathy index (Kyte-Doolittle)
# Source: https://en.wikipedia.org/wiki/Hydrophobicity_scales
HYDROPATHY_INDEX = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

class DNASequence:
    """
    Represents a DNA sequence and provides methods for common operations.
    """
    def __init__(self, sequence: str):
        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string.")
        self.sequence = sequence.upper()
        if not all(base in 'ATGC' for base in self.sequence):
            raise ValueError("DNA sequence can only contain A, T, G, C.")

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        return f"DNASequence('{self.sequence[:20]}...')" if len(self.sequence) > 20 else f"DNASequence('{self.sequence}')"

    def gc_content(self) -> float:
        """
        Calculates the GC content of the DNA sequence.
        Returns:
            float: The percentage of G and C bases in the sequence.
        """
        g_count = self.sequence.count('G')
        c_count = self.sequence.count('C')
        total_bases = len(self.sequence)
        return (g_count + c_count) / total_bases * 100 if total_bases > 0 else 0.0

    def nucleotide_frequency(self) -> dict:
        """
        Calculates the frequency of each nucleotide (A, T, G, C) in the DNA sequence.
        Returns:
            dict: A dictionary where keys are nucleotides and values are their counts.
        """
        return collections.Counter(self.sequence)

    def reverse_complement(self) -> 'DNASequence':
        """
        Returns the reverse complement of the DNA sequence.
        Returns:
            DNASequence: A new DNASequence object representing the reverse complement.
        """
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        reversed_seq = self.sequence[::-1]
        complement_seq = "".join([complement_map.get(base, base) for base in reversed_seq]) # Use .get for robustness
        return DNASequence(complement_seq)

    def find_motif(self, motif: str, allow_ambiguity: bool = False) -> list[int]:
        """
        Finds all occurrences of a given motif in the DNA sequence.
        Args:
            motif (str): The motif sequence to search for. Can contain 'N' if allow_ambiguity is True.
            allow_ambiguity (bool): If True, 'N' in motif matches any base (A, T, G, C).
        Returns:
            list[int]: A list of start indices where the motif is found.
        Raises:
            ValueError: If motif contains invalid characters for DNA or if 'N' is used without allow_ambiguity.
        """
        if not isinstance(motif, str):
            raise TypeError("Motif must be a string.")
        
        upper_motif = motif.upper()
        if not all(base in 'ATGC N' for base in upper_motif):
            raise ValueError("Motif can only contain A, T, G, C, or N (if ambiguity allowed).")
        if 'N' in upper_motif and not allow_ambiguity:
            raise ValueError("Motif contains 'N' but allow_ambiguity is False. Set allow_ambiguity=True to match any base.")

        indices = []
        n_motif = len(upper_motif)
        if n_motif == 0:
            return []

        for i in range(len(self.sequence) - n_motif + 1):
            match = True
            for j in range(n_motif):
                seq_base = self.sequence[i+j]
                motif_base = upper_motif[j]
                if motif_base == 'N':
                    # 'N' matches any nucleotide
                    if seq_base not in 'ATGC': # Ensure sequence base is valid
                        match = False
                        break
                elif seq_base != motif_base:
                    match = False
                    break
            if match:
                indices.append(i)
        return indices

    def transcribe(self) -> 'RNASequence':
        """
        Transcribes the DNA sequence into an RNA sequence (T -> U).
        Returns:
            RNASequence: A new RNASequence object.
        """
        rna_seq_str = self.sequence.replace('T', 'U')
        return RNASequence(rna_seq_str)

    def find_orfs(self, min_protein_length: int = 10, genetic_code: dict = None) -> list['ProteinSequence']:
        """
        Finds open reading frames (ORFs) and translates them into protein sequences.
        An ORF starts with 'AUG' and ends with a stop codon ('UAA', 'UAG', 'UGA').
        Args:
            min_protein_length (int): Minimum length of the translated protein to be considered an ORF.
            genetic_code (dict): Custom genetic code mapping codons to amino acids. Defaults to STANDARD_GENETIC_CODE.
        Returns:
            list[ProteinSequence]: A list of ProteinSequence objects found.
        """
        rna_seq = self.transcribe()
        proteins = []
        gc = genetic_code if genetic_code else STANDARD_GENETIC_CODE
        start_codon = 'AUG'
        stop_codons = ['UAA', 'UAG', 'UGA']

        for frame in range(3):
            current_protein_bases = []
            in_orf = False
            for i in range(frame, len(rna_seq.sequence) - 2, 3):
                codon = rna_seq.sequence[i:i+3]
                
                if not in_orf:
                    if codon == start_codon:
                        in_orf = True
                        current_protein_bases = [gc.get(codon, 'X')] # Start with 'M'
                elif in_orf:
                    if codon in stop_codons:
                        if len(current_protein_bases) >= min_protein_length:
                            proteins.append(ProteinSequence("".join(current_protein_bases)))
                        in_orf = False
                        current_protein_bases = [] # Reset for next ORF
                    else:
                        amino_acid = gc.get(codon, 'X')
                        current_protein_bases.append(amino_acid)
            
            # If an ORF runs to the end of the sequence without a stop codon
            if in_orf and len(current_protein_bases) >= min_protein_length:
                proteins.append(ProteinSequence("".join(current_protein_bases)))

        return proteins


class RNASequence:
    """
    Represents an RNA sequence and provides methods for common operations.
    """
    def __init__(self, sequence: str):
        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string.")
        self.sequence = sequence.upper()
        if not all(base in 'AUGC' for base in self.sequence):
            raise ValueError("RNA sequence can only contain A, U, G, C.")

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        return f"RNASequence('{self.sequence[:20]}...')" if len(self.sequence) > 20 else f"RNASequence('{self.sequence}')"

    def gc_content(self) -> float:
        """
        Calculates the GC content of the RNA sequence.
        Returns:
            float: The percentage of G and C bases in the sequence.
        """
        g_count = self.sequence.count('G')
        c_count = self.sequence.count('C')
        total_bases = len(self.sequence)
        return (g_count + c_count) / total_bases * 100 if total_bases > 0 else 0.0

    def nucleotide_frequency(self) -> dict:
        """
        Calculates the frequency of each nucleotide (A, U, G, C) in the RNA sequence.
        Returns:
            dict: A dictionary where keys are nucleotides and values are their counts.
        """
        return collections.Counter(self.sequence)

    def translate(self, genetic_code: dict = None) -> list['ProteinSequence']:
        """
        Translates the RNA sequence into protein sequences considering all three reading frames.
        Args:
            genetic_code (dict): Custom genetic code mapping codons to amino acids. Defaults to STANDARD_GENETIC_CODE.
        Returns:
            list[ProteinSequence]: A list of potential ProteinSequence objects (one for each reading frame).
        """
        gc = genetic_code if genetic_code else STANDARD_GENETIC_CODE
        protein_sequences = []
        for frame in range(3):
            protein = []
            for i in range(frame, len(self.sequence) - 2, 3):
                codon = self.sequence[i:i+3]
                amino_acid = gc.get(codon, 'X') # 'X' for unknown/invalid codon
                protein.append(amino_acid)
                if amino_acid == '*': # Stop codon
                    break
            protein_sequences.append(ProteinSequence("".join(protein)))
        return protein_sequences


class ProteinSequence:
    """
    Represents a protein sequence and provides methods for analysis.
    """
    # Allowed 20 standard amino acids + Stop codon '*' + 'X' for unknown/gap
    ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWXY*"

    def __init__(self, sequence: str):
        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string.")
        self.sequence = sequence.upper()
        if not all(aa in self.ALLOWED_AMINO_ACIDS for aa in self.sequence):
            invalid_aas = [aa for aa in self.sequence if aa not in self.ALLOWED_AMINO_ACIDS]
            raise ValueError(f"Protein sequence contains invalid amino acids: {set(invalid_aas)}. Allowed: {self.ALLOWED_AMINO_ACIDS}")

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        return f"ProteinSequence('{self.sequence[:20]}...')" if len(self.sequence) > 20 else f"ProteinSequence('{self.sequence}')"

    def amino_acid_composition(self) -> dict:
        """
        Calculates the frequency of each amino acid in the protein sequence.
        Returns:
            dict: A dictionary where keys are amino acids and values are their counts.
        """
        return collections.Counter(self.sequence)

    def molecular_weight(self) -> float:
        """
        Estimates the molecular weight of the protein sequence (in Daltons).
        Based on average molecular weights of amino acid residues.
        Returns:
            float: The estimated molecular weight in Daltons.
        Notes:
            This is an approximation. Actual molecular weight depends on specific
            isotopes and post-translational modifications. A water molecule (18.01528 Da)
            is added to account for the N-terminal H and C-terminal OH.
        """
        mw = 0.0
        for aa in self.sequence:
            if aa in AMINO_ACID_WEIGHTS:
                mw += AMINO_ACID_WEIGHTS[aa]
            elif aa == '*': # Stop codon, typically not part of final protein weight
                continue
            elif aa == 'X': # Unknown amino acid, cannot add weight
                continue
            else:
                # This should ideally be caught by __init__ validation, but for robustness
                raise ValueError(f"Unknown amino acid '{aa}' in sequence for molecular weight calculation.")

        if len(self.sequence) > 0:
            mw += 18.01528 # Add weight of H2O for N-terminal H and C-terminal OH

        return mw

    def hydrophobicity(self, window_size: int = 9) -> list[float]:
        """
        Calculates the Kyte-Doolittle hydrophobicity for the protein sequence using a sliding window.
        Args:
            window_size (int): The size of the sliding window. Must be an odd number.
        Returns:
            list[float]: A list of average hydrophobicity values for each window.
        Raises:
            ValueError: If window_size is not odd or if it's larger than the sequence length.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        if window_size > len(self.sequence):
            raise ValueError("Window size cannot be larger than the sequence length.")
        
        scores = []
        for i in range(len(self.sequence) - window_size + 1):
            window = self.sequence[i : i + window_size]
            window_score = sum(HYDROPATHY_INDEX.get(aa, 0.0) for aa in window if aa in HYDROPATHY_INDEX)
            scores.append(window_score / window_size)
        return scores

    def find_pattern(self, pattern: str) -> list[int]:
        """
        Finds all occurrences of a given amino acid pattern in the protein sequence.
        Args:
            pattern (str): The amino acid pattern to search for.
        Returns:
            list[int]: A list of start indices where the pattern is found.
        Raises:
            ValueError: If pattern contains characters not allowed in protein sequences.
        """
        if not isinstance(pattern, str):
            raise TypeError("Pattern must be a string.")
        if not all(aa in self.ALLOWED_AMINO_ACIDS for aa in pattern.upper()):
            raise ValueError(f"Pattern contains invalid amino acids. Allowed: {self.ALLOWED_AMINO_ACIDS}")
        
        indices = []
        upper_pattern = pattern.upper()
        n_pattern = len(upper_pattern)
        if n_pattern == 0:
            return []

        for i in range(len(self.sequence) - n_pattern + 1):
            if self.sequence[i:i + n_pattern] == upper_pattern:
                indices.append(i)
        return indices


class GeneExpressionData:
    """
    Handles gene expression data, typically from RNA-seq or microarray experiments.
    Assumes data is in a pandas DataFrame, where rows are genes and columns are samples.
    """
    def __init__(self, data_path: str = None, data_df: pd.DataFrame = None, sep: str = '\t', index_col: int = 0):
        """
        Initializes by loading data from a file or directly from a pandas DataFrame.
        Args:
            data_path (str, optional): Path to the data file (e.g., CSV, TSV).
            data_df (pd.DataFrame, optional): A pandas DataFrame containing expression data.
            sep (str): Delimiter for the file (default: tab). Used if data_path is provided.
            index_col (int): Column to use as index (e.g., gene names, default: 0). Used if data_path is provided.
        Raises:
            ValueError: If neither data_path nor data_df is provided, or if data is not numeric.
            FileNotFoundError: If data_path is provided but the file does not exist.
        """
        if data_path:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            try:
                self.data = pd.read_csv(data_path, sep=sep, index_col=index_col)
            except Exception as e:
                raise ValueError(f"Error loading data from {data_path}: {e}")
        elif data_df is not None:
            if not isinstance(data_df, pd.DataFrame):
                raise TypeError("data_df must be a pandas DataFrame.")
            self.data = data_df
        else:
            raise ValueError("Either 'data_path' or 'data_df' must be provided.")

        if not pd.api.types.is_numeric_dtype(self.data.values):
            # Check if non-numeric columns exist (excluding the index)
            non_numeric_cols = self.data.select_dtypes(exclude=np.number).columns.tolist()
            if non_numeric_cols:
                raise ValueError(f"Expression data contains non-numeric values in columns: {non_numeric_cols}")

    def __str__(self) -> str:
        return f"Gene Expression Data (Genes: {self.data.shape[0]}, Samples: {self.data.shape[1]})"

    def __repr__(self) -> str:
        return f"GeneExpressionData(genes={self.data.shape[0]}, samples={self.data.shape[1]})"

    def get_gene_names(self) -> list[str]:
        """
        Returns a list of gene names (index of the DataFrame).
        Returns:
            list[str]: A list of gene identifiers.
        """
        return self.data.index.tolist()

    def get_sample_names(self) -> list[str]:
        """
        Returns a list of sample names (columns of the DataFrame).
        Returns:
            list[str]: A list of sample identifiers.
        """
        return self.data.columns.tolist()

    def get_expression_values(self, identifier: str, is_gene: bool = True) -> pd.Series:
        """
        Returns expression values for a specific gene or sample.
        Args:
            identifier (str): The name of the gene or sample.
            is_gene (bool): If True, treats identifier as a gene name (row). If False, as a sample name (column).
        Returns:
            pd.Series: A pandas Series containing the expression values.
        Raises:
            ValueError: If the identifier is not found.
        """
        if is_gene:
            if identifier not in self.data.index:
                raise ValueError(f"Gene '{identifier}' not found in data.")
            return self.data.loc[identifier]
        else:
            if identifier not in self.data.columns:
                raise ValueError(f"Sample '{identifier}' not found in data.")
            return self.data[identifier]

    def calculate_statistics(self, axis: int = 0) -> pd.DataFrame:
        """
        Calculates basic statistics (mean, median, std dev) for samples (axis=0) or genes (axis=1).
        Args:
            axis (int): 0 for statistics per sample (columns), 1 for statistics per gene (rows).
        Returns:
            pd.DataFrame: A DataFrame with 'Mean', 'Median', 'Std Dev' as columns.
        Raises:
            ValueError: If axis is not 0 or 1.
        """
        if axis not in [0, 1]:
            raise ValueError("Axis must be 0 (samples) or 1 (genes).")
        
        stats_df = pd.DataFrame({
            'Mean': self.data.mean(axis=axis),
            'Median': self.data.median(axis=axis),
            'Std Dev': self.data.std(axis=axis)
        })
        return stats_df

    def normalize_data(self, method: str = 'minmax') -> 'GeneExpressionData':
        """
        Normalizes the expression data using the specified method.
        Args:
            method (str): Normalization method. Options: 'minmax', 'zscore'.
        Returns:
            GeneExpressionData: A new GeneExpressionData object with normalized data.
        Raises:
            ValueError: If an unknown normalization method is specified.
        """
        normalized_df = self.data.copy()

        if method == 'minmax':
            min_vals = normalized_df.min()
            max_vals = normalized_df.max()
            range_vals = max_vals - min_vals
            # Avoid division by zero for columns with constant values
            normalized_df = (normalized_df - min_vals) / range_vals.replace(0, 1)
        elif method == 'zscore':
            mean_vals = normalized_df.mean()
            std_vals = normalized_df.std()
            # Avoid division by zero for columns with zero std dev
            normalized_df = (normalized_df - mean_vals) / std_vals.replace(0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}. Choose 'minmax' or 'zscore'.")
        
        return GeneExpressionData(data_df=normalized_df)

    def calculate_log2_fold_change(self, group1_samples: list[str], group2_samples: list[str], pseudocount: float = 1.0) -> pd.Series:
        """
        Calculates the log2 fold change between two groups of samples.
        Args:
            group1_samples (list[str]): List of sample names for group 1.
            group2_samples (list[str]): List of sample names for group 2.
            pseudocount (float): A small value added to expression values to avoid log(0) errors.
        Returns:
            pd.Series: A Series with log2 fold change for each gene.
        Raises:
            ValueError: If any sample name is not found or if a group is empty.
        """
        if not all(s in self.data.columns for s in group1_samples):
            raise ValueError("One or more sample names in group1_samples not found.")
        if not all(s in self.data.columns for s in group2_samples):
            raise ValueError("One or more sample names in group2_samples not found.")
        if not group1_samples or not group2_samples:
            raise ValueError("Both group1_samples and group2_samples cannot be empty.")

        avg_group1 = (self.data[group1_samples].mean(axis=1) + pseudocount)
        avg_group2 = (self.data[group2_samples].mean(axis=1) + pseudocount)
        
        # Ensure no division by zero after adding pseudocount
        if (avg_group1 == 0).any() or (avg_group2 == 0).any():
             # This should ideally not happen with pseudocount, but as a safeguard:
             raise RuntimeError("Encountered zero average expression even after adding pseudocount. Check data or pseudocount.")

        log2fc = np.log2(avg_group1 / avg_group2)
        return log2fc

    def filter_genes(self, min_expression: float = 0.0, max_expression: float = float('inf'), min_samples_expressed: int = 0) -> 'GeneExpressionData':
        """
        Filters genes based on their expression levels across samples.
        Args:
            min_expression (float): Minimum average expression for a gene to be kept.
            max_expression (float): Maximum average expression for a gene to be kept.
            min_samples_expressed (int): Minimum number of samples a gene must be expressed in (above 0).
        Returns:
            GeneExpressionData: A new GeneExpressionData object with filtered genes.
        """
        filtered_df = self.data.copy()

        # Filter by average expression
        if min_expression > 0.0 or max_expression < float('inf'):
            gene_means = filtered_df.mean(axis=1)
            filtered_df = filtered_df[(gene_means >= min_expression) & (gene_means <= max_expression)]
        
        # Filter by number of samples expressed
        if min_samples_expressed > 0:
            num_expressed_samples = (filtered_df > 0).sum(axis=1)
            filtered_df = filtered_df[num_expressed_samples >= min_samples_expressed]

        return GeneExpressionData(data_df=filtered_df)


# --- Utility Functions for File I/O ---

def read_fasta(filepath: str) -> dict[str, DNASequence | RNASequence | ProteinSequence]:
    """
    Reads sequences from a FASTA file. Automatically determines sequence type (DNA/RNA/Protein).
    Args:
        filepath (str): Path to the FASTA file.
    Returns:
        dict[str, Union[DNASequence, RNASequence, ProteinSequence]]: A dictionary where keys are sequence headers
            (without the '>') and values are DNASequence, RNASequence, or ProteinSequence objects.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is malformed or sequence type cannot be determined.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FASTA file not found at {filepath}")

    sequences = {}
    current_id = None
    current_sequence_lines = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id and current_sequence_lines:
                    full_sequence = "".join(current_sequence_lines)
                    sequences[current_id] = _determine_and_create_sequence_object(full_sequence)
                current_id = line[1:].split(' ')[0] # Take first part of header as ID
                current_sequence_lines = []
            else:
                current_sequence_lines.append(line)
        
        # Add the last sequence
        if current_id and current_sequence_lines:
            full_sequence = "".join(current_sequence_lines)
            sequences[current_id] = _determine_and_create_sequence_object(full_sequence)
    
    if not sequences:
        raise ValueError(f"No sequences found in FASTA file: {filepath}. Is it empty or malformed?")
    
    return sequences

def _determine_and_create_sequence_object(sequence_str: str) -> DNASequence | RNASequence | ProteinSequence:
    """Helper to determine sequence type and create appropriate object."""
    upper_seq = sequence_str.upper()
    
    # Check for DNA (contains T, no U)
    if 'U' not in upper_seq and all(base in 'ATGC' for base in upper_seq):
        return DNASequence(upper_seq)
    # Check for RNA (contains U, no T)
    elif 'T' not in upper_seq and all(base in 'AUGC' for base in upper_seq):
        return RNASequence(upper_seq)
    # Check for Protein (contains common amino acid codes, may contain 'X' or '*')
    elif all(aa in ProteinSequence.ALLOWED_AMINO_ACIDS for aa in upper_seq):
        return ProteinSequence(upper_seq)
    else:
        raise ValueError(f"Could not determine sequence type for: {upper_seq[:30]}...")


def write_fasta(filepath: str, sequences: dict[str, DNASequence | RNASequence | ProteinSequence]) -> None:
    """
    Writes a dictionary of sequence objects to a FASTA file.
    Args:
        filepath (str): Path to the output FASTA file.
        sequences (dict): A dictionary where keys are sequence headers and values are
                          DNASequence, RNASequence, or ProteinSequence objects.
    Raises:
        ValueError: If sequences is empty or contains non-sequence objects.
    """
    if not sequences:
        raise ValueError("No sequences provided to write to FASTA file.")
    
    with open(filepath, 'w') as f:
        for header, seq_obj in sequences.items():
            if not isinstance(seq_obj, (DNASequence, RNASequence, ProteinSequence)):
                raise ValueError(f"Value for header '{header}' is not a valid sequence object.")
            f.write(f">{header}\n")
            # Write sequence in chunks for readability (e.g., 60 characters per line)
            for i in range(0, len(seq_obj.sequence), 60):
                f.write(seq_obj.sequence[i:i+60] + '\n')


def write_expression_data(filepath: str, expr_data: GeneExpressionData, sep: str = '\t') -> None:
    """
    Writes gene expression data to a file (CSV/TSV).
    Args:
        filepath (str): Path to the output file.
        expr_data (GeneExpressionData): The GeneExpressionData object to write.
        sep (str): Delimiter for the output file (default: tab).
    Raises:
        TypeError: If expr_data is not a GeneExpressionData object.
        Exception: For other file writing errors.
    """
    if not isinstance(expr_data, GeneExpressionData):
        raise TypeError("expr_data must be an instance of GeneExpressionData.")
    try:
        expr_data.data.to_csv(filepath, sep=sep, index_label=expr_data.data.index.name or 'GeneID')
    except Exception as e:
        raise Exception(f"Error writing expression data to {filepath}: {e}")

# --- Example Usage (demonstrates library functions) ---

if __name__ == "__main__":
    print("--- Bioinformatics Tools Library Demonstration ---")

    # --- DNASequence Examples ---
    print("\n--- DNA Sequence Analysis ---")
    dna_seq1 = DNASequence("ATGCGTACGATCGATCGATGCATGCATGCATGCGATCGATCGATCGATCGA")
    print(f"DNA Sequence: {dna_seq1}")
    print(f"Length: {len(dna_seq1)}")
    print(f"GC Content: {dna_seq1.gc_content():.2f}%")
    print(f"Nucleotide Frequencies: {dna_seq1.nucleotide_frequency()}")
    print(f"Reverse Complement: {dna_seq1.reverse_complement()}")

    motif1 = "ATCG"
    indices1 = dna_seq1.find_motif(motif1)
    print(f"Motif '{motif1}' found at indices: {indices1}")

    dna_seq2 = DNASequence("AGCTAGCNATCGNAGCT")
    motif2 = "AGCN"
    indices2 = dna_seq2.find_motif(motif2, allow_ambiguity=True)
    print(f"DNA Sequence for ambiguous motif: {dna_seq2}")
    print(f"Ambiguous Motif '{motif2}' found at indices: {indices2}")

    try:
        dna_seq2.find_motif("AGCN", allow_ambiguity=False)
    except ValueError as e:
        print(f"Expected error for ambiguous motif without flag: {e}")

    # --- RNASequence Examples ---
    print("\n--- RNA Sequence Analysis ---")
    rna_seq1 = dna_seq1.transcribe()
    print(f"Transcribed RNA: {rna_seq1}")
    print(f"RNA GC Content: {rna_seq1.gc_content():.2f}%")
    print(f"RNA Nucleotide Frequencies: {rna_seq1.nucleotide_frequency()}")

    # --- ProteinSequence Examples ---
    print("\n--- Protein Sequence Analysis ---")
    
    # Translate from RNA
    protein_options = rna_seq1.translate()
    for i, prot in enumerate(protein_options):
        print(f"Protein from Frame {i+1}: {prot}")
        print(f"  Amino Acid Composition: {prot.amino_acid_composition()}")
        print(f"  Estimated Molecular Weight: {prot.molecular_weight():.2f} Da")
        pattern_to_find = "L"
        pattern_indices = prot.find_pattern(pattern_to_find)
        if pattern_indices:
            print(f"  Pattern '{pattern_to_find}' found at indices: {pattern_indices}")
        else:
            print(f"  Pattern '{pattern_to_find}' not found.")
        hydrophobicity_scores = prot.hydrophobicity(window_size=5)
        print(f"  Hydrophobicity (5-mer): {hydrophobicity_scores[:5]}...") # print first 5 scores

    # Find ORFs from DNA
    dna_with_orf = DNASequence("ATGCGAUGCCUGAACUAGCUAAGCUGAUGCGUAG") # Contains an ORF: AUG-CCU-GAA-CUA-GCU-AAG-CUG-AUG-CGU-AG (ends with UAG)
    print(f"\nDNA sequence for ORF finding: {dna_with_orf}")
    orfs = dna_with_orf.find_orfs(min_protein_length=5)
    print(f"Found {len(orfs)} ORFs:")
    for i, orf_prot in enumerate(orfs):
        print(f"  ORF {i+1}: {orf_prot}")


    # --- GeneExpressionData Examples ---
    print("\n--- Gene Expression Data Analysis ---")
    # Create a dummy expression data file for demonstration
    dummy_data = """Gene\tSample_A1\tSample_A2\tSample_B1\tSample_B2\tSample_C1
Gene1\t100\t120\t50\t60\t150
Gene2\t50\t60\t100\t110\t55
Gene3\t200\t210\t20\t25\t205
Gene4\t5\t8\t1000\t1200\t7
Gene5\t0\t0\t0\t0\t10
Gene6\t10\t12\t11\t13\t10
"""
    dummy_filepath = "dummy_expression_data.tsv"
    with open(dummy_filepath, "w") as f:
        f.write(dummy_data)

    try:
        expr_data = GeneExpressionData(data_path=dummy_filepath)
        print(expr_data)
        print("\nSample Statistics:")
        print(expr_data.calculate_statistics(axis=0))
        print("\nGene Statistics:")
        print(expr_data.calculate_statistics(axis=1))

        normalized_minmax = expr_data.normalize_data(method='minmax')
        print("\nMin-Max Normalized Data (first 5 rows):\n", normalized_minmax.data.head())

        normalized_zscore = expr_data.normalize_data(method='zscore')
        print("\nZ-score Normalized Data (first 5 rows):\n", normalized_zscore.data.head())

        # Differential Expression Example
        group1 = ['Sample_A1', 'Sample_A2']
        group2 = ['Sample_B1', 'Sample_B2']
        log2fc_results = expr_data.calculate_log2_fold_change(group1, group2)
        print(f"\nLog2 Fold Change ({', '.join(group1)} vs {', '.join(group2)}):\n", log2fc_results)

        # Gene Filtering Example
        filtered_expr = expr_data.filter_genes(min_expression=20.0, min_samples_expressed=2)
        print("\nFiltered Expression Data (Min Avg Expr > 20, Min 2 samples expressed):\n", filtered_expr.data)

        # Write expression data example
        output_expr_filepath = "output_expression_data.tsv"
        write_expression_data(output_expr_filepath, normalized_minmax)
        print(f"\nNormalized data written to {output_expr_filepath}")

    except Exception as e:
        print(f"An error occurred during expression data demo: {e}")
    finally:
        if os.path.exists(dummy_filepath):
            os.remove(dummy_filepath)
        if os.path.exists(output_expr_filepath):
            os.remove(output_expr_filepath)

    # --- FASTA File I/O Examples ---
    print("\n--- FASTA File I/O Demonstration ---")
    fasta_dna_content = """>gene1
ATGCGTA
TGCATGC
>gene2
AAAAACCCC
GGGGTTTTT
"""
    fasta_protein_content = """>protA
MFPQAWS
KLYP*
>protB
VGDIE
"""

    dna_fasta_filepath = "dna_sequences.fasta"
    protein_fasta_filepath = "protein_sequences.fasta"

    with open(dna_fasta_filepath, "w") as f:
        f.write(fasta_dna_content)
    with open(protein_fasta_filepath, "w") as f:
        f.write(fasta_protein_content)

    try:
        dna_sequences_from_file = read_fasta(dna_fasta_filepath)
        print(f"\nRead DNA sequences from {dna_fasta_filepath}:")
        for header, seq_obj in dna_sequences_from_file.items():
            print(f"  {header}: {seq_obj} ({type(seq_obj).__name__})")
            print(f"    GC Content: {seq_obj.gc_content():.2f}%")

        protein_sequences_from_file = read_fasta(protein_fasta_filepath)
        print(f"\nRead Protein sequences from {protein_fasta_filepath}:")
        for header, seq_obj in protein_sequences_from_file.items():
            print(f"  {header}: {seq_obj} ({type(seq_obj).__name__})")
            print(f"    MW: {seq_obj.molecular_weight():.2f} Da")

        # Write sequences back to a new FASTA file
        combined_sequences_for_writing = {**dna_sequences_from_file, **protein_sequences_from_file}
        output_fasta_filepath = "combined_sequences_output.fasta"
        write_fasta(output_fasta_filepath, combined_sequences_for_writing)
        print(f"\nCombined sequences written to {output_fasta_filepath}")

    except Exception as e:
        print(f"An error occurred during FASTA I/O demo: {e}")
    finally:
        if os.path.exists(dna_fasta_filepath):
            os.remove(dna_fasta_filepath)
        if os.path.exists(protein_fasta_filepath):
            os.remove(protein_fasta_filepath)
        if os.path.exists(output_fasta_filepath):
            os.remove(output_fasta_filepath)

    print("\n--- Demonstration Complete ---")
