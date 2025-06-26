import os
import json
import torch
import dgl
import networkx as nx

def load_graphs_from_json(folder):
    graphs = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), 'r') as f:
                data = json.load(f)
            g = nx.Graph()
            g.add_nodes_from(range(data['n']))
            g.add_edges_from(data['edges'])
            dgl_g = dgl.from_networkx(g)
            dgl_g.ndata['feat'] = torch.tensor(data['features'], dtype=torch.float32)
            graphs.append({
                'id': data['id'],
                'graph': dgl_g
            })
    return graphs

def load_graph_pairs(graphs):
    pairs = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            pairs.append((graphs[i]['graph'], graphs[j]['graph'], torch.tensor([abs(graphs[i]['graph'].number_of_nodes() - graphs[j]['graph'].number_of_nodes())], dtype=torch.float32)))
    return pairs