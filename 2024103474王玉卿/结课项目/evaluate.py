import os 
import json 
import argparse 
from typing import List, Dict, Set, Tuple 

from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim 
import torch 
import numpy as np 

from base import Schema 
from log import get_logger 
# 获取日志记录器，用于记录程序运行信息
logger = get_logger('root')


def read_reference_schema(filename:str, embedding_model: SentenceTransformer):
    """
    读取参考模式文件，并计算事件嵌入、时间边集合和层次边集合。

    :param filename: 参考模式文件的路径
    :param embedding_model: 用于计算事件描述嵌入的模型
    :return: 事件列表、事件嵌入字典、时间边集合和层次边集合
    """
    with open(filename,'r') as f:
        # 加载参考模式文件的 JSON 数据
        ref_schema = json.load(f)
    
    event_list = ref_schema['events']
    event_embs = {} 
    temporal_edge_set = set() 
    hier_edge_set = set() 
    id_set = set() 

    # 预先计算所有事件的嵌入
    for evt in event_list:
        eid = evt['@id']
        if 'description' not in evt or evt['description'] == '': 
            # 若事件没有描述信息，抛出异常
            raise ValueError(f'{filename}: {eid} does not have an description')
        # 计算事件描述的嵌入，并转换为 PyTorch 张量
        new_embed = embedding_model.encode(evt['description'], convert_to_tensor=True) # type: torch.FloatTensor
        event_embs[eid] = new_embed 
        id_set.add(eid)

    # 收集时间边和层次边
    for evt in event_list:
        eid = evt['@id']
        if 'outlinks' in evt:
            for other_eid in evt['outlinks']:
                if other_eid in id_set:
                    # 添加时间边到集合中
                    temporal_edge_set.add((eid, other_eid))
        if 'children' in evt:
            for other_eid in evt['children']:
                if other_eid in id_set: 
                    # 添加层次边到集合中
                    hier_edge_set.add((eid, other_eid))
        
    return event_list, event_embs, temporal_edge_set, hier_edge_set

def create_emb_matrix(event_list:List, event_embs:Dict):
    '''
    根据事件列表和事件嵌入字典创建嵌入矩阵。

    :param event_list: 事件 ID 列表
    :param event_embs: 事件嵌入字典
    :return: 事件嵌入矩阵
    '''
    tensor_list = []
    for evt in event_list:
        tensor_list.append(event_embs[evt])

    # 沿着第 0 维堆叠张量列表，创建嵌入矩阵
    emb_matrix = torch.stack(tensor_list, dim=0)
    return emb_matrix 


def compute_edge_metrics(ref_edges:Set[Tuple], gen_edges: Set[Tuple], gen2ref_assignment:Dict, ref2gen_assignment:Dict):
    """
    计算边的精确率、召回率和 F1 值。

    :param ref_edges: 参考边集合
    :param gen_edges: 生成边集合
    :param gen2ref_assignment: 生成事件到参考事件的映射字典
    :param ref2gen_assignment: 参考事件到生成事件的映射字典
    :return: 精确率、召回率、F1 值、参考边数量和生成边数量
    """
    # 计算边的 F1 值
    # 获取匹配的参考事件和生成事件
    matched_ref = set(gen2ref_assignment.values())
    matched_gen = set(gen2ref_assignment.keys()) 
    correct_n =0 

    # 获取匹配的参考边和生成边
    matched_ref_edges = set((x,y) for (x,y) in ref_edges if (x in matched_ref and y in matched_ref))
    matched_gen_edges = set((x,y) for (x,y) in gen_edges if (x in matched_gen and y in matched_gen))

    for ref_pair in matched_ref_edges:
        # 将参考边映射到生成边
        mapped_pair = (ref2gen_assignment[ref_pair[0]], ref2gen_assignment[ref_pair[1]])
        if mapped_pair in matched_gen_edges:
            # 若映射后的边在生成边集合中，正确边数量加 1
            correct_n += 1

    if len(matched_ref_edges) ==0: 
        recall =0
    else:
        # 计算召回率
        recall = correct_n/len(matched_ref_edges)
    if len(matched_gen_edges) ==0:
        prec =0
    else:
        # 计算精确率
        prec = correct_n/ len(matched_gen_edges)
    
    if correct_n == 0:
        f1 = 0.0
    else:
        # 计算 F1 值
        f1 = 2/ (1/prec + 1/recall)

    return prec, recall, f1, len(matched_ref_edges), len(matched_gen_edges)

    



if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser('Evaluation of the schema against human ground truth. All the schemas in the reference dir will be used.')
    parser.set_defaults(use_retrieval=False)

    # 添加命令行参数
    parser.add_argument('--reference_schema_dir',type=str, default='schemalib/phase2b/curated')
    parser.add_argument('--schema_dir', type=str, default='outputs/schema_phase2b_Jan14')
    # parser.add_argument('--scenario', type=str, default='chemical spill')
    args = parser.parse_args() 

    # 加载 SentenceTransformer 模型
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = {} 
    assignments= {} 

    for filename in os.listdir(args.reference_schema_dir):
        # 获取文件名和扩展名
        scenario, ext = os.path.splitext(filename)
        if ext!= '.json': continue 
        # 将场景名转换为小写
        scenario = scenario.lower()
        if scenario == 'chemical_spill': continue # 跳过用于上下文学习的场景

        # if scenario == 'coup': continue # 跳过有 XOR 门问题的场景


        # 读取参考模式文件
        ref_event_list, ref_embs, ref_temp, ref_hier = read_reference_schema(
            os.path.join(args.reference_schema_dir, filename), sbert_model)

        # 创建参考事件索引到 ID 和 ID 到索引的映射字典
        ref_idx2id = {idx: evt['@id'] for idx, evt in enumerate(ref_event_list)}
        ref_id2idx = {evt['@id']: idx for idx, evt in enumerate(ref_event_list)} 

        ref_N = len(ref_event_list)
        logger.info(f'The reference schema contains {ref_N} events')

        # 从文件中加载生成的模式
        s= Schema.from_file(args.schema_dir, scenario, embedding_model=sbert_model)

        N = len(s.events) 
        logger.info(f'The generated schema contains {N} events')

        results[scenario] = {}

        # 构建分配矩阵
        emb_matrix = create_emb_matrix(list(sorted(s.events.keys())), s.event_embs)
        ref_event_ids= [evt['@id'] for evt in ref_event_list]
        ref_emb_matrix = create_emb_matrix(ref_event_ids, ref_embs)

        # 计算相似度矩阵
        sim_matrix = cos_sim(emb_matrix, ref_emb_matrix)
        # 将相似度矩阵转换为成本矩阵
        w = sim_matrix.cpu().numpy()
        # w = np.zeros((N, ref_N), dtype=np.int64) # cost matrix
        # 使用线性和分配算法进行事件匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix=w, maximize=True)

        # 创建生成事件到参考事件的映射字典
        gen2ref_assignment = {x:ref_idx2id[y] for x,y in zip(row_ind, col_ind)}
        ref2gen_assignment = {v:k for k,v in gen2ref_assignment.items()}
        # 事件 ID 到 @id 的映射

        assignments[scenario] = []
        for k,v in gen2ref_assignment.items():
            ref_evt = ref_event_list[ref_id2idx[v]]
            assignments[scenario].append({
                'ref_id': ref_evt['@id'],
                'ref_event': ref_evt['name'],
                'ref_event_description': ref_evt['description'],
                'gen_id': s.events[k].id,
                'gen_event': s.events[k].name,
                'gen_event_description': s.events[k].description
            })


        # 行索引和列索引的长度等于 min(N, ref_N)
        score = w[row_ind, col_ind].sum()
        # 计算事件精确率
        event_prec = score / N
        # 计算事件召回率
        event_recall = score / ref_N

        # 计算事件 F1 值
        event_f1 = 2/ (1/event_prec + 1/event_recall)

        print(f'Prec: {event_prec:.3f} Recall: {event_recall:.3f} F1: {event_f1:.3f}')
        results[scenario]['event'] = {
            'prec': event_prec,
            'recall': event_recall,
            'f1': event_f1,
            'ref_n': ref_N,
            'gen_n': N 
        }


        # 收集生成模式中的边
        temp_edges = set()
        hier_edges = set() 
        for eid, evt in s.events.items():
            for other_eid in evt.children:
                hier_edges.add((eid, other_eid))
            
            for other_eid in evt.after:
                temp_edges.add((eid, other_eid))

        # 计算时间边的指标
        temp_prec, temp_recall, temp_f1, ref_n, gen_n = compute_edge_metrics(ref_temp, temp_edges, gen2ref_assignment, ref2gen_assignment)

        print(f'Temporal edge Prec: {temp_prec:.3f} Recall: {temp_recall:.3f} F1: {temp_f1:.3f}')
        results[scenario]['temporal'] = {
            'prec': temp_prec,
            'recall': temp_recall,
            'f1': temp_f1,
            'ref_n': ref_n,
            'gen_n': gen_n
        }

        # 计算层次边的指标
        hier_prec, hier_recall, hier_f1, ref_n, gen_n = compute_edge_metrics(ref_hier, hier_edges, gen2ref_assignment, ref2gen_assignment)

        results[scenario]['hierarchical'] = {
            'prec': hier_prec,
            'recall': hier_recall,
            'f1': hier_f1,
    # 将事件分配结果保存到文件
    # 将评估结果保存到文件
            # 计算平均值
    # 对所有场景进行平均
            'ref_n': ref_n,
            'gen_n': gen_n 
        }

        # print(f'Hierarchical edge Prec: {hier_prec:.3f} Recall: {hier_recall:.3f} F1: {hier_f1:.3f}')



    # average over all scenarios 
    summarized = {
        'event':{
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
        'temporal': {
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
        'hierarchical':
         {
            'prec': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ref_n': 0.0,
            'gen_n': 0.0
        },
    }
    
    for aspect in ['event','temporal','hierarchical']:
        for metric in ['prec','recall','f1','ref_n','gen_n']:
            val = []
            for scenario in results:
                val.append(results[scenario][aspect][metric])
            avg_val = np.mean(np.array(val)) 
            summarized[aspect][metric] = avg_val 
        

    results['total'] = summarized

    with open(os.path.join(args.schema_dir,'eval.json'),'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.schema_dir,'assignments.json'),'w') as f:
        json.dump(assignments, f, indent=2) 

