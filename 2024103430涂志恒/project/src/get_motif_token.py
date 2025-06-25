# coding=utf-8
# get_motif_token

import json
from collections import defaultdict
from rdkit.Chem import BRICS
from tqdm import tqdm
import pandas as pd
from get_mol import get_mol
import re


def clean_brics_smiles(smi):
    # 替换所有数字标记为[*]
    return re.sub(r'\[\d+\*]', '[*]', smi)


def build_motif_vocabulary(chebi_list, output_json):
    vocab = defaultdict(int)
    for chebi_id in tqdm(chebi_list, desc="Scanning motifs"):
        mol = get_mol(chebi_id)
        if mol is None:
            continue
        try:
            frag_smiles = BRICS.BRICSDecompose(mol)
            for smi in frag_smiles:
                smi = clean_brics_smiles(smi)
                vocab[smi] += 1
        except Exception as e:
            print(f"Error processing {chebi_id}: {str(e)}")
    sorted_motifs = sorted(vocab.items(), key=lambda x: -x[1])
    sorted_motifs = filter(lambda x: x[1] > 1, sorted_motifs)
    motif_vocab = {
        "[PADDING]": -1,
        "[GLOBAL]": 0,
        "[UNK]": 1,
        **{smi: idx + 2 for idx, (smi, _) in enumerate(sorted_motifs)}
    }
    with open(output_json, 'w') as f:
        json.dump(motif_vocab, f, indent=2)
    print(f"Vocabulary saved to {output_json}")
    print(f"Total motifs: {len(motif_vocab)}")
    return motif_vocab


if __name__ == "__main__":
    df = pd.read_excel('data/enzyme_substrate_train.xlsx')
    build_motif_vocabulary(
        chebi_list=list(set(df['molecule ID'].tolist())),
        output_json="data/motif_token.json"
    )
