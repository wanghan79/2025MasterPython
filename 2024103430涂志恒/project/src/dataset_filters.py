# coding=utf-8
# dataset_filters

import pandas as pd
import pickle


def filter_valid_molecules(input_path, output_path):
    df = pd.read_excel(input_path)
    valid_data = []
    failed_list = []
    for row in df.to_dict('records'):
        try:
            pickle.load(open(f'sub_input/{row["molecule ID"].replace(":", "_")}.pkl', 'rb'))
            pickle.load(open(f'esm2_feat/{row["Uniprot ID"]}.pkl', 'rb'))
            pickle.load(open(f'protein_graphs/{row["Uniprot ID"]}.pkl', 'rb'))
            valid_data.append(row)
        except:
            failed_list.append(row)
            continue
    valid_df = pd.DataFrame(valid_data)
    valid_df.to_excel(output_path, index=False)
    return len(valid_data), len(failed_list)


if __name__ == '__main__':
    train_valid_count, train_failed = filter_valid_molecules(
        'data/enzyme_substrate_train.xlsx',
        'data/enzyme_substrate_train_filtereds.xlsx'
    )
    test_valid_count, test_failed = filter_valid_molecules(
        'data/enzyme_substrate_test.xlsx',
        'data/enzyme_substrate_test_filtereds.xlsx'
    )
    print(f"训练集: 保留 {train_valid_count}/{train_valid_count + train_failed} 条数据")
    print(f"测试集: 保留 {test_valid_count}/{test_valid_count + test_failed} 条数据")
