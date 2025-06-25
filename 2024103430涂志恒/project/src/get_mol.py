# coding=utf-8
# get_mol

import pandas as pd
from rdkit import Chem
import os
from os.path import join

CURRENT_DIR = os.getcwd()
df_chebi_to_inchi = pd.read_csv(join(CURRENT_DIR, "data", "chebiID_to_inchi.tsv"), sep="\t")
mol_folder = join(CURRENT_DIR, "mol-files")


def get_mol(met_ID):
    is_CHEBI_ID = (met_ID[0:5] == "CHEBI")
    is_InChI = (met_ID[0:5] == "InChI")
    if is_CHEBI_ID:
        try:
            ID = int(met_ID.split(" ")[0].split(":")[-1])
            Inchi = list(df_chebi_to_inchi["Inchi"].loc[df_chebi_to_inchi["ChEBI"] == float(ID)])[0]
            mol = Chem.inchi.MolFromInchi(Inchi)
        except:
            mol = None
    elif is_InChI:
        try:
            mol = Chem.inchi.MolFromInchi(met_ID)
        except:
            mol = None
    else:
        try:
            mol = Chem.MolFromMolFile(mol_folder + "\\" + met_ID + '.mol')
        except:
            mol = None
    return mol


if __name__ == '__main__':
    print(get_mol('CHEBI:57344'))
