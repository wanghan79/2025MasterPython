import settings
from pandas.core.frame import DataFrame
import re
from torch.utils import data
import pandas as pd

class Dataset(data.Dataset):
    """ A class implementing :class:`torch.utils.data.Dataset`.

    Dataset subclasses the abstract class :class:`torch.utils.data.Dataset`. The class overrides
    ``__len__``, ``__getitem__``, ``__contains__``, ``__str__``, ``__eq__`` and ``__init__``.

    Dataset is a two-dimensional immutable, potentially heterogeneous tabular data structure with
    labeled axes (rows and columns).

    Args:
        rows (list of dict): Construct a two-dimensional tabular data structure from rows.

    Attributes:
        columns (set of string): Set of column names.
    """

    def __init__(self, rows):
        self.columns = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        """
        Get a column or row from the dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
        Returns:
            :class:`list` or :class:`dict`: List of column values or a dict representing a row
        """
        # Given an column string return list of column values.
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError('Key not in columns.')
            return [row[key] if key in row else None for row in self.rows]
        # Given an row integer return a object of row values.
        elif isinstance(key, (int, slice)):
            return self.rows[key]
        else:
            raise TypeError('Invalid argument type.')

    def __setitem__(self, key, item):
        """
        Set a column or row for a dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
            item (list or dict): Column or rows to set in the dataset.
        """
        if isinstance(key, str):
            column = item
            self.columns.add(key)
            if len(column) > len(self.rows):
                for i, value in enumerate(column):
                    if i < len(self.rows):
                        self.rows[i][key] = value
                    else:
                        self.rows.append({key: value})
            else:
                for i, row in enumerate(self.rows):
                    if i < len(column):
                        self.rows[i][key] = column[i]
                    else:
                        self.rows[i][key] = None
        elif isinstance(key, slice):
            rows = item
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('Row must be a dict.')
                self.columns.update(row.keys())
            self.rows[key] = rows
        elif isinstance(key, int):
            row = item
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
            self.rows[key] = row
        else:
            raise TypeError('Invalid argument type.')

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.columns

    def __str__(self):
        return str(pd.DataFrame(self.rows))

    # def __eq__(self, other):
    #     return self.columns == other.columns and self.rows == other.rows

    def __add__(self, other):
        return Dataset(self.rows + other.rows)

class PPIDataset():
    """
    Loads the Dataset from the csv files passed to the parser.
    """
    def  __init__(self) -> None:
        return

    def collate_lists(self, seqA: list, seqB: list, labels: list, protA_ids: list, protB_ids: list, ncbi_geneA_id: list, ncbi_geneB_id: list) -> list:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seqA)):
            collated_dataset.append({
                "seqA": str(seqA[i]), 
                "seqB": str(seqB[i]), 
                "label": str(labels[i]),
                "protA_ids": str(protA_ids[i]).split(","), 
                "protB_ids": str(protB_ids[i]).split(","), 
                "ncbi_geneA_id": str(ncbi_geneA_id[i]), 
                "ncbi_geneB_id": str(ncbi_geneB_id[i]), 
            })
        return collated_dataset

    def _retrieve_dataframe(self, path) -> DataFrame:
        column_names = [
            "class", 
            'sequenceA', 'sequenceB', 
            "protein_A_ids", "protein_B_ids", 
            "ncbi_gene_A_id", "ncbi_gene_B_id"
        ] 
        df: DataFrame = pd.read_csv(path, sep = "\t", names=column_names, header=0) #type:ignore
        return df

    def calculate_stat(self,path):
        df = self._retrieve_dataframe(path)
        self.nSamples_dic = df['class'].value_counts()

    def load_predict_dataset(self, path):
        column_names = [
            "interaction", "probability",
            'receptor_protein_id',
            "receptor_protein_label",
            'receptor_protein_name',
            'receptor_protein_sequence',
            'capsid_protein_sequence',
        ] 
        df: DataFrame = pd.read_csv(path, sep = "\t", names=column_names, header=0) #type:ignore

        interactions = list(df['interaction'])
        probabilities = list(df['probability'])
        receptor_protein_ids = list(df['receptor_protein_id'])
        receptor_protein_labels = list(df['receptor_protein_label'])
        receptor_protein_names = list(df['receptor_protein_name'])
        receptor_protein_seqs = list(df['receptor_protein_sequence'])
        capsid_protein_sequences = list(df['capsid_protein_sequence']) 

        # Make sure there is a space between every token, and map rarely amino acids
        receptor_protein_seqs = [" ".join("".join(sample.split())) for sample in receptor_protein_seqs]
        receptor_protein_seqs = [re.sub(r"[UZOB]", "X", sample) for sample in receptor_protein_seqs]
        
        capsid_protein_sequences = [" ".join("".join(sample.split())) for sample in capsid_protein_sequences]
        capsid_protein_sequences = [re.sub(r"[UZOB]", "X", sample) for sample in capsid_protein_sequences]

        assert len(receptor_protein_seqs) == len(interactions)
        assert len(capsid_protein_sequences) == len(interactions)
        assert len(receptor_protein_ids) == len(interactions)
        assert len(receptor_protein_names) == len(interactions)
        assert len(receptor_protein_labels) == len(interactions)
        assert len(probabilities) == len(interactions)

        collated_dataset = []
        for i in range(len(interactions)):
            collated_dataset.append({
                "label": str(interactions[i]),
                "probability": str(probabilities[i]), 
                "receptor_protein_id": str(receptor_protein_ids[i]), 
                "receptor_protein_label": str(receptor_protein_labels[i]), 
                "receptor_protein_name": str(receptor_protein_names[i]), 
                "seqA": str(receptor_protein_seqs[i]), 
                "seqB": str(capsid_protein_sequences[i]), 
            })

        return Dataset(collated_dataset)

    def load_dataset(self, path):
        df = self._retrieve_dataframe(path)

        labels = list(df['class'])
        seqA = list(df['sequenceA'])
        seqB = list(df['sequenceB'])
        protA_ids = list(df['protein_A_ids']) # TODO: Rename to protein_A_ids
        protB_ids = list(df['protein_B_ids']) # TODO: Rename to protein_A_ids
        ncbi_geneA_id = list(df['ncbi_gene_A_id'])
        ncbi_geneB_id = list(df['ncbi_gene_B_id'])

        # Make sure there is a space between every token, and map rarely amino acids
        seqA = [" ".join("".join(sample.split())) for sample in seqA]
        seqA = [re.sub(r"[UZOB]", "X", sample) for sample in seqA]
        
        seqB = [" ".join("".join(sample.split())) for sample in seqB]
        seqB = [re.sub(r"[UZOB]", "X", sample) for sample in seqB]

        assert len(seqA) == len(labels)
        assert len(seqB) == len(labels)
        assert len(protA_ids) == len(labels)
        assert len(protB_ids) == len(labels)
        
        return Dataset(self.collate_lists(seqA, seqB, labels, protA_ids, protB_ids, ncbi_geneA_id, ncbi_geneB_id))


