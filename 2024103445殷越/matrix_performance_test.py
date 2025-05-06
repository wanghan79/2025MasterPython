import time
import random
import copy

def create_list_matrix(size):
    """Create a size x size matrix using list"""
    return [[0 for _ in range(size)] for _ in range(size)]

def create_tuple_matrix(size):
    """Create a size x size matrix using tuple"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def modify_list_matrix(matrix, iterations):
    """Modify list matrix one element at a time"""
    size = len(matrix)
    start_time = time.time()
    
    for _ in range(iterations):
        # Randomly select position
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        # Modify value
        matrix[i][j] = 1
    
    end_time = time.time()
    return end_time - start_time

def modify_tuple_matrix(matrix, iterations):
    """Modify tuple matrix one element at a time"""
    size = len(matrix)
    start_time = time.time()
    
    for _ in range(iterations):
        # Randomly select position
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        # For tuple, we need to create new tuples for the modification
        temp_row = list(matrix[i])
        temp_row[j] = 1
        matrix = matrix[:i] + (tuple(temp_row),) + matrix[i+1:]
    
    end_time = time.time()
    return end_time - start_time

def main():
    # Parameters
    size = 10000  # 10000 x 10000 matrix
    iterations = 10000  # 10000 modifications
    
    print(f"Creating {size}x{size} matrices...")
    
    # Create matrices
    list_matrix = create_list_matrix(size)
    tuple_matrix = create_tuple_matrix(size)
    
    print("\nStarting performance test...")
    print(f"Performing {iterations} modifications on each matrix type...")
    
    # Test list performance
    print("\nTesting List performance...")
    list_time = modify_list_matrix(list_matrix, iterations)
    print(f"List modification time: {list_time:.2f} seconds")
    
    # Test tuple performance
    print("\nTesting Tuple performance...")
    tuple_time = modify_tuple_matrix(tuple_matrix, iterations)
    print(f"Tuple modification time: {tuple_time:.2f} seconds")
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"List was {tuple_time/list_time:.2f}x faster than Tuple")

if __name__ == "__main__":
    main() 