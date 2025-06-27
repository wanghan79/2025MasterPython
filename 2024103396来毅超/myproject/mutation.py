from utils import LLM_generate_algorithm

class MutationModule:
    def __init__(self, task, prompt_mutation, extra_prompt):
        self.task = task
        self.prompt_mutation = prompt_mutation
        self.extra_prompt = extra_prompt
    
    def mutate_algorithm(self, algorithm):
        all_prompt = self.prompt_mutation + algorithm + "\n" + self.extra_prompt
        mutate_algorithm, cost = LLM_generate_algorithm(all_prompt, stochasticity=1.5)
        return mutate_algorithm, cost

if __name__ == '__main__':
    print('Mutation Module')
