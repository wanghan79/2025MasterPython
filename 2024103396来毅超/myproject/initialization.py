import random
from utils import LLM_generate_algorithm

class InitializationModule:
    def __init__(self, task, prompt_init, prompt_code_request, extra_prompt, handmade=False):
        self.all_prompt = task + prompt_init + prompt_code_request + extra_prompt
        self.handmade = handmade


    # 生成多个候选算法
    def generate_initial_algorithms(self, count):
        if not self.handmade:
            algorithms = [LLM_generate_algorithm(self.all_prompt) for i in range(count)]
            cost = sum(cost for _, cost in algorithms)
        else:
            algorithms = self.handmade_algorithms()
            cost = 0
        return algorithms[:count], cost

    def handmade_algorithms(self):
        algorithms = [
        """import numpy as np
def score_nodes(adjacency_matrix):
    degrees = np.sum(adjacency_matrix, axis=1)
    scored_nodes = {}
    for node_id, degree in enumerate(degrees):
        scored_nodes[node_id] = degree
    return scored_nodes""",
        """import networkx as nx
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    betweenness = nx.betweenness_centrality(G)
    return betweenness""",
        """import numpy as np
def score_nodes(adjacency_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, max_eigenvalue_index]
    normalized_eigenvector = eigenvector / np.sum(eigenvector)
    scored_nodes = {}
    for node_id, centrality_score in enumerate(normalized_eigenvector):
        scored_nodes[node_id] = centrality_score
    return scored_nodes""",
        """import numpy as np
def score_nodes(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    clustering_coefficients = np.zeros(num_nodes)
    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node] == 1)[0]
        num_neighbors = len(neighbors)
        if num_neighbors < 2:
            clustering_coefficients[node] = 0.0
        else:
            num_connected_pairs = 0
            for i in range(num_neighbors):
                for j in range(i + 1, num_neighbors):
                    if adjacency_matrix[neighbors[i], neighbors[j]] == 1:
                        num_connected_pairs += 1
            clustering_coefficients[node] = (2.0 * num_connected_pairs) / (num_neighbors * (num_neighbors - 1))
    scored_nodes = {node_id: clustering_coefficients[node_id] for node_id in range(num_nodes)}
    return scored_nodes""",
        """import numpy as np
def score_nodes(adjacency_matrix):
    node_degree_map = np.sum(adjacency_matrix, axis=0)
    num_nodes = adjacency_matrix.shape[0]
    node_ci = np.zeros(num_nodes)
    scored_nodes = {}
    for node in range(num_nodes):
        ci = 0
        neighbors = np.where(adjacency_matrix[node] == 1)[0]
        for neighbor in neighbors:
            ci += (node_degree_map[neighbor] - 1)
        node_ci[node] = ci * (node_degree_map[node] - 1)
        scored_nodes[node] = node_ci[node]
    return scored_nodes""",
    """import networkx as nx
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    scored_nodes = nx.pagerank(G)
    return scored_nodes""",
    """import networkx as nx
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    scored_nodes = nx.percolation_centrality(G)
    return scored_nodes""",
    """import networkx as nx
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    scored_nodes = nx.harmonic_centrality(G)
    return scored_nodes""",
    """import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    LCC = G.subgraph(max(nx.connected_components(G), key=len))
    ii = {v: i for i, v in enumerate(LCC.nodes())}
    L = nx.normalized_laplacian_matrix(LCC)
    eigenvalues, eigenvectors = eigsh(L.astype(np.float32), k=2, which='SM', maxiter=1000 * L.shape[0])
    Fiedler = eigenvectors[:, 1]
    H = nx.Graph([(u, v) for u, v in LCC.edges() if Fiedler[ii[u]] * Fiedler[ii[v]] <= 0.0])
    for v in H.nodes():
        H.nodes[v]['weight'] = 1.0 / H.degree(v)
    cover = list(nx.algorithms.approximation.min_weighted_vertex_cover(H, weight='weight'))
    max_degree = max([G.degree(v) for v in G.nodes() if v not in cover])
    min_weight = min(H.nodes[v]['weight'] for v in H.nodes())
    scored_nodes = {v: H.nodes[v]['weight'] if v in cover else G.degree(v) / max_degree * min_weight for v in G.nodes()}
    return scored_nodes""",
    """import networkx as nx
def score_nodes(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    score1 = {node_id: 0.0 for node_id in G.nodes()}
    for u, v in G.edges():
        degree_u = G.degree(u)
        degree_v = G.degree(v)
        score1[u] += 1.0 / degree_u if degree_u != 0 else 0.0
        score1[v] += 1.0 / degree_v if degree_v != 0 else 0.0
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    weight_degree = 0.2
    weight_betweenness = 0.3
    weight_pagerank = 0.5
    scored_nodes = {}
    for node_id in G.nodes():
        score = (weight_degree * score1.get(node_id, 0) +
                 weight_betweenness * betweenness[node_id] +
                 weight_pagerank * pagerank[node_id])
        scored_nodes[node_id] = score
    return scored_nodes"""
        ]
        return algorithms 

if __name__ == '__main__':
    print('Initialization Module')