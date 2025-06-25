# coding=utf-8
# get_pdb

import requests
import urllib3
import pandas as pd
from tqdm import tqdm
import os

urllib3.disable_warnings()


def read_file():
    """读取Excel文件并返回Uniprot ID列表"""
    df1 = pd.read_excel('enzyme_substrate_train.xlsx')
    df2 = pd.read_excel('enzyme_substrate_test.xlsx')
    return df1['Uniprot ID'].tolist() + df2['Uniprot ID'].tolist()


def get_best_alphafold_pdb(uniprot_id, save_dir='pdb'):
    """
    获取最佳可用的AlphaFold PDB文件
    按v4 > v3 > v2 > v1的顺序尝试下载
    """
    versions = ['v4', 'v3', 'v2', 'v1']
    base_url = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_{}.pdb'
    for version in versions:
        url = base_url.format(uniprot_id, version)
        for _ in range(3):
            try:
                response = requests.get(url, headers=headers, verify=False, timeout=(10, 10))
                break
            except:
                response = '<?xml'
        if response.status_code == 200 and not response.text.startswith('<?xml'):
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            # 保存文件
            save_path = os.path.join(save_dir, f'{uniprot_id}.pdb')
            with open(save_path, 'w') as f:
                f.write(response.text)
            return version  # 返回成功下载的版本
    return None  # 所有版本都不可用


if __name__ == '__main__':
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'
    }
    # 获取所有唯一的Uniprot ID
    uniprot_ids = set(read_file())
    print(f"总共需要处理的Uniprot ID数量: {len(uniprot_ids)}")
    # 用于记录结果的字典
    results = {
        'success': [],
        'failed': []
    }
    # 使用tqdm显示进度条
    for uniprot_id in tqdm(uniprot_ids, desc="下载PDB文件"):
        version = get_best_alphafold_pdb(uniprot_id)
        if version:
            results['success'].append((uniprot_id, version))
        else:
            results['failed'].append(uniprot_id)
    print(results)
