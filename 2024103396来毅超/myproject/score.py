import random
import multiprocessing
import time
from utils import sample_by_degree_distribution, calculate_anc, calculate_anc_gcc, sorted_by_value
from scipy.sparse import load_npz

class ScoringModule:

    def __init__(self, file_path, ratio, calculate_type='pc'):
        self.edge_matrix = load_npz(file_path).toarray()
        if len(self.edge_matrix) > 5000:
            self.edge_matrix = sample_by_degree_distribution(self.edge_matrix, 0.1)
        self.number_of_removed_nodes = int(ratio * 0.01 * len(self.edge_matrix))
        self.metric = calculate_type
        print(f'removed number: {self.number_of_removed_nodes}, metric: {self.metric}')

    def score_nodes_with_timeout(self, algorithm, timeout=60):
        def run_algorithm(result_queue):
            try:
                exec(algorithm, globals_dict)
                result_dict = globals_dict['score_nodes'](self.edge_matrix)
                result = sorted_by_value(result_dict)
                if self.metric == 'pc':
                    score = 1 - calculate_anc(self.edge_matrix, result[:self.number_of_removed_nodes])
                elif self.metric == 'gcc':
                    score = 1 - calculate_anc_gcc(self.edge_matrix, result[:self.number_of_removed_nodes])
                else:
                    score = 0
                result_queue.put(score)
            except Exception as e:
                print("This code can not be evaluated!")
                result_queue.put(0)

        globals_dict = {}
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_algorithm, args=(result_queue,))
        
        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return 0
        else:
            return result_queue.get()

    def evaluate_algorithm(self, algorithm):
        score = self.score_nodes_with_timeout(algorithm, timeout=60)
        return score

if __name__ == '__main__':
    print('Score Module')
