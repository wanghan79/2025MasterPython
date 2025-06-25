# coding=utf-8
# get_esm2_features

import torch
import esm
import pickle
import pandas as pd
import os


def get_esm2_features(seq, return_per_residue=False, layer=33):
    data = [("enzyme", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
    torch.cuda.empty_cache()
    token_embeddings = results["representations"][layer].cpu()
    residue_embeddings = token_embeddings[0, 1:-1, :]
    sequence_embedding = residue_embeddings.mean(dim=0)
    output = {"sequence_embedding": sequence_embedding.numpy()}
    if return_per_residue:
        output["per_residue_embedding"] = residue_embeddings.numpy()
    return output


if __name__ == '__main__':
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    df1 = pd.read_excel('data/enzyme_substrate_train_filtereds.xlsx')
    df2 = pd.read_excel('data/enzyme_substrate_test_filtereds.xlsx')
    df = pd.concat([df1, df2], ignore_index=True)
    uniprots = []
    for idx, row in df.iterrows():
        uniprot_id = row['Uniprot ID']
        sequence = row['Sequence']
        if uniprot_id in uniprots or os.path.exists(f'esm2_feat/{uniprot_id}.pkl'):
            continue
        uniprots.append(uniprot_id)
        try:
            features = get_esm2_features(sequence, return_per_residue=False)
            esm2_feature = features['sequence_embedding']
            with open(f'esm2_feat/{uniprot_id}.pkl', 'wb') as f:
                pickle.dump(esm2_feature, f)
            print(idx, esm2_feature)
        except:
            print('drop', idx)
