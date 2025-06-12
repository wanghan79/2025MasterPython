import time
import logging
from datetime import datetime


def setup_logger():
    """配置日志记录器"""
    log_filename = f'matrix_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def create_matrix(size, use_tuple=False):
    """创建指定大小的矩阵"""
    if use_tuple:
        return [tuple([0] * size) for _ in range(size)]
    return [[0] * size for _ in range(size)]


def modify_matrix(matrix, count, use_tuple=False):
    """修改矩阵对角线上的元素"""
    for i in range(count):
        if use_tuple:
            row = list(matrix[i])
            row[i] = 1
            matrix[i] = tuple(row)
        else:
            matrix[i][i] = 1


def main():
    logger = setup_logger()
    matrix_size = 10000
    modify_count = 10000

    # 测试列表矩阵
    start = time.perf_counter()
    list_matrix = create_matrix(matrix_size)
    list_create_time = time.perf_counter() - start
    logger.info(f"列表矩阵创建时间: {list_create_time:.4f}秒")

    # 测试元组矩阵
    start = time.perf_counter()
    tuple_matrix = create_matrix(matrix_size, use_tuple=True)
    tuple_create_time = time.perf_counter() - start
    logger.info(f"元组矩阵创建时间: {tuple_create_time:.4f}秒")

    # 修改列表矩阵
    start = time.perf_counter()
    modify_matrix(list_matrix, modify_count)
    list_modify_time = time.perf_counter() - start
    logger.info(f"列表修改时间: {list_modify_time:.4f}秒")

    # 修改元组矩阵
    start = time.perf_counter()
    modify_matrix(tuple_matrix, modify_count, use_tuple=True)
    tuple_modify_time = time.perf_counter() - start
    logger.info(f"元组修改时间: {tuple_modify_time:.4f}秒")

    # 结果比较
    logger.info(f"元组修改时间是列表的 {tuple_modify_time / list_modify_time:.1f} 倍")


if __name__ == "__main__":
    main()
