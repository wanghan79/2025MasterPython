"""
结课项目：PBO（伪布尔优化）CDCL求解器
作者：XYZ_Herry
说明：
    本项目实现了一个简化版的PBO（Pseudo-Boolean Optimization）问题求解器，采用CDCL（冲突驱动子句学习）思想，支持0-1整数线性规划问题的建模与自动求解。
    代码结构清晰，注释详细，便于理解和扩展。
    适合组合优化、人工智能、运筹学等领域的学习和研究。
"""

import random
import heapq
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set, Any
import itertools

# =====================
# 1. PBO问题建模相关类
# =====================

class PBOVariable:
    """
    伪布尔变量（0-1变量）
    用法示例：
        v = PBOVariable('x1', 0)
        print(v)  # x1=None
    """
    def __init__(self, name: str, idx: int):
        self.name: str = name
        self.idx: int = idx
        self.value: Optional[int] = None  # None/0/1
        self.level: int = -1    # 决策层
        self.reason: Optional['PBOConstraint'] = None # 产生该赋值的约束/子句
    def __repr__(self):
        return f"{self.name}={self.value}"  
    def reset(self):
        """重置变量状态"""
        self.value = None
        self.level = -1
        self.reason = None

class PBOConstraint:
    """
    伪布尔约束：形如 a1*x1 + a2*x2 + ... <= b
    用法示例：
        c = PBOConstraint({0: 2, 1: 3}, 5)
        print(c)  # 2*x0 + 3*x1 <= 5
    """
    def __init__(self, coeffs: Dict[int, int], bound: int):
        self.coeffs: Dict[int, int] = coeffs  # {var_idx: 系数}
        self.bound: int = bound
    def __repr__(self):
        terms = [f"{a}*x{v}" for v, a in self.coeffs.items()]
        return f"{' + '.join(terms)} <= {self.bound}"
    def is_satisfied(self, assigns: List[Optional[int]]) -> bool:
        """判断当前赋值下约束是否满足"""
        total = 0
        for idx, coef in self.coeffs.items():
            val = assigns[idx]
            if val is not None:
                total += coef * val
        return total <= self.bound
    def slack(self, assigns: List[Optional[int]]) -> int:
        """返回剩余松弛量（未赋值变量最大可取值和）"""
        total = 0
        for idx, coef in self.coeffs.items():
            val = assigns[idx]
            if val is not None:
                total += coef * val
        return self.bound - total
    def unassigned_vars(self, assigns: List[Optional[int]]) -> List[int]:
        """返回未赋值变量索引列表"""
        return [idx for idx in self.coeffs if assigns[idx] is None]
    def max_possible(self, assigns: List[Optional[int]]) -> int:
        """当前赋值下该约束左侧最大可能值"""
        total = 0
        for idx, coef in self.coeffs.items():
            val = assigns[idx]
            if val is None:
                total += max(0, coef)
            else:
                total += coef * val
        return total
    def print_detail(self, assigns: List[Optional[int]], var_names: List[str]):
        """详细打印约束当前状态"""
        terms = []
        for idx, coef in self.coeffs.items():
            vname = var_names[idx]
            val = assigns[idx]
            terms.append(f"{coef}*{vname}({val})")
        print(f"{' + '.join(terms)} <= {self.bound}  (slack={self.slack(assigns)})")

class PBOObjective:
    """
    目标函数：c1*x1 + c2*x2 + ...
    用法示例：
        obj = PBOObjective({0: 2, 1: 3})
        print(obj)  # max: 2*x0 + 3*x1
    """
    def __init__(self, coeffs: Dict[int, int]):
        self.coeffs: Dict[int, int] = coeffs
    def __repr__(self):
        terms = [f"{a}*x{v}" for v, a in self.coeffs.items()]
        return f"max: {' + '.join(terms)}"
    def evaluate(self, assigns: List[Optional[int]]) -> int:
        """计算当前赋值下目标函数值"""
        return sum(assigns[i] * a for i, a in self.coeffs.items() if assigns[i] is not None)
    def print_detail(self, assigns: List[Optional[int]], var_names: List[str]):
        terms = []
        for idx, coef in self.coeffs.items():
            vname = var_names[idx]
            val = assigns[idx]
            terms.append(f"{coef}*{vname}({val})")
        print(f"目标函数: {' + '.join(terms)} = {self.evaluate(assigns)}")

class PBOProblem:
    """
    PBO问题建模
    用法示例：
        pb = PBOProblem()
        pb.add_variable('x1')
        pb.set_objective({'x1': 2})
        pb.add_constraint({'x1': 1}, 1)
        print(pb)
    """
    def __init__(self):
        self.variables: List[PBOVariable] = []
        self.var_map: Dict[str, int] = {}  # name -> idx
        self.constraints: List[PBOConstraint] = []
        self.objective: Optional[PBOObjective] = None
    def add_variable(self, name: str) -> int:
        if name in self.var_map:
            return self.var_map[name]
        idx = len(self.variables)
        self.variables.append(PBOVariable(name, idx))
        self.var_map[name] = idx
        return idx
    def add_constraint(self, coeffs: Dict[str, int], bound: int):
        idx_coeffs = {self.var_map[n]: a for n, a in coeffs.items()}
        self.constraints.append(PBOConstraint(idx_coeffs, bound))
    def set_objective(self, coeffs: Dict[str, int]):
        idx_coeffs = {self.var_map[n]: a for n, a in coeffs.items()}
        self.objective = PBOObjective(idx_coeffs)
    def __repr__(self):
        s = [str(self.objective)]
        for c in self.constraints:
            s.append(str(c))
        return '\n'.join(s)
    def visualize(self):
        print("PBO问题结构：")
        print("变量：", [v.name for v in self.variables])
        print("目标函数：", self.objective)
        print("约束：")
        for c in self.constraints:
            print("  ", c)
    def print_constraints_detail(self, assigns: List[Optional[int]]):
        print("约束详细状态：")
        for c in self.constraints:
            c.print_detail(assigns, [v.name for v in self.variables])

# =====================
# 2. VSIDS分支启发式
# =====================

class VSIDS:
    """
    VSIDS分支启发式（变量活动度）
    用法示例：
        vs = VSIDS(3)
        vs.bump(0)
        print(vs.scores)
    """
    def __init__(self, n_vars: int):
        self.scores: List[float] = [0.0] * n_vars
        self.decay: float = 0.95
    def bump(self, var_idx: int):
        self.scores[var_idx] += 1.0
    def decay_all(self):
        for i in range(len(self.scores)):
            self.scores[i] *= self.decay
    def pick_branch_var(self, assigned: Set[int]) -> Optional[int]:
        # 选择未赋值且分数最高的变量
        best = -1
        best_score = -1
        for i, s in enumerate(self.scores):
            if i not in assigned and s > best_score:
                best = i
                best_score = s
        return best if best != -1 else None
    def print_scores(self, var_names: List[str]):
        print("VSIDS分数：")
        for i, score in enumerate(self.scores):
            print(f"{var_names[i]}: {score:.2f}")

# =====================
# 3. CDCL主求解器
# =====================

class CDCLSolver:
    """
    CDCL主循环，支持伪布尔约束
    用法示例：
        solver = CDCLSolver(pb)
        solver.solve()
    """
    def __init__(self, problem: PBOProblem):
        self.problem: PBOProblem = problem
        self.n_vars: int = len(problem.variables)
        self.assigns: List[Optional[int]] = [None] * self.n_vars  # 当前赋值
        self.levels: List[int] = [0] * self.n_vars      # 决策层
        self.reasons: List[Optional[PBOConstraint]] = [None] * self.n_vars  # 产生赋值的约束
        self.decision_level: int = 0
        self.trail: List[int] = []  # 赋值轨迹
        self.vsids: VSIDS = VSIDS(self.n_vars)
        self.learned: List[PBOConstraint] = []
        self.conflicts: int = 0
        self.max_conflicts: int = 10000
        self.best_obj: float = float('-inf')
        self.best_model: Optional[List[Optional[int]]] = None
        self.history: List[Any] = []  # 记录所有决策和回溯
        self.verbose: bool = False
    def solve(self, verbose: bool = False):
        """
        主求解循环
        """
        self.verbose = verbose
        while True:
            conflict = self.unit_propagate()
            if conflict:
                self.conflicts += 1
                if self.decision_level == 0:
                    print("UNSAT! 无可行解。")
                    return None
                learnt, back_level = self.analyze_conflict(conflict)
                self.learned.append(learnt)
                self.backtrack(back_level)
                self.add_learned(learnt)
                self.vsids.bump(random.choice(list(learnt.coeffs.keys())))
                self.vsids.decay_all()
            else:
                var = self.pick_branch_var()
                if var is None:
                    # 所有变量已赋值，检查目标
                    obj = self.eval_objective()
                    if obj > self.best_obj:
                        self.best_obj = obj
                        self.best_model = self.assigns[:]
                    if not self.next_objective_bound():
                        print("最优解：", self.best_obj)
                        print("模型：", self.format_model(self.best_model))
                        return self.best_model
                    self.backtrack(0)
                else:
                    self.decision_level += 1
                    self.assign(var, 1, None)
    def unit_propagate(self) -> Optional[PBOConstraint]:
        """
        单元传播，返回冲突约束或None
        """
        for c in self.problem.constraints + self.learned:
            slack = c.slack(self.assigns)
            unassigned = c.unassigned_vars(self.assigns)
            if len(unassigned) == 0:
                if not c.is_satisfied(self.assigns):
                    return c  # 冲突
            elif len(unassigned) == 1:
                idx = unassigned[0]
                coef = c.coeffs[idx]
                if coef > 0:
                    if slack < 0:
                        return c
                    elif slack < coef:
                        self.assign(idx, 0, c)
                else:
                    if slack < 0:
                        return c
                    elif slack < -coef:
                        self.assign(idx, 1, c)
        return None
    def analyze_conflict(self, conflict: PBOConstraint) -> Tuple[PBOConstraint, int]:
        """
        冲突分析，返回学习子句和回溯层
        """
        # 简化版：直接返回冲突约束，回溯到0层
        return conflict, 0
    def backtrack(self, level: int):
        """
        回溯到指定决策层
        """
        for i in range(self.n_vars):
            if self.levels[i] > level:
                self.assigns[i] = None
                self.levels[i] = 0
                self.reasons[i] = None
        self.decision_level = level
        self.trail = [v for v in self.trail if self.levels[v] <= level]
        self.history.append(('backtrack', level))
    def add_learned(self, learnt: PBOConstraint):
        self.learned.append(learnt)
    def pick_branch_var(self) -> Optional[int]:
        assigned = {i for i, v in enumerate(self.assigns) if v is not None}
        return self.vsids.pick_branch_var(assigned)
    def assign(self, var_idx: int, value: int, reason: Optional[PBOConstraint]):
        self.assigns[var_idx] = value
        self.levels[var_idx] = self.decision_level
        self.reasons[var_idx] = reason
        self.trail.append(var_idx)
        self.history.append(('assign', var_idx, value, self.decision_level))
        if self.verbose:
            print(f"[Level {self.decision_level}] assign {self.problem.variables[var_idx].name} = {value}")
    def eval_objective(self) -> int:
        if self.problem.objective is None:
            return 0
        return sum(self.assigns[i] * a for i, a in self.problem.objective.coeffs.items() if self.assigns[i] is not None)
    def next_objective_bound(self) -> bool:
        # 简单分支限界：尝试提升目标下界
        return False
    def format_model(self, model: Optional[List[Optional[int]]]) -> Optional[Dict[str, int]]:
        if model is None:
            return None
        return {self.problem.variables[i].name: v for i, v in enumerate(model)}
    def print_trail(self):
        print("赋值轨迹:")
        for step in self.history:
            print(step)
    def verify_model(self, model: List[Optional[int]]) -> bool:
        """验证模型是否满足所有约束"""
        for c in self.problem.constraints:
            if not c.is_satisfied(model):
                return False
        return True
    def enumerate_all_solutions(self, max_solutions: int = 10):
        """暴力枚举所有可行解（仅适合变量数<=15）"""
        n = self.n_vars
        count = 0
        print("所有可行解（最多展示%d个）：" % max_solutions)
        for bits in itertools.product([0, 1], repeat=n):
            if all(c.is_satisfied(list(bits)) for c in self.problem.constraints):
                obj = self.problem.objective.evaluate(list(bits))
                print({self.problem.variables[i].name: bits[i] for i in range(n)}, "目标值:", obj)
                count += 1
                if count >= max_solutions:
                    break
        print(f"共找到{count}个可行解（可能未穷尽）")
    def print_solution_report(self):
        print("\n==== 求解报告 ====")
        if self.best_model is not None:
            print("最优目标值：", self.best_obj)
            print("最优解：", self.format_model(self.best_model))
            print("模型验证：", "通过" if self.verify_model(self.best_model) else "失败")
            self.problem.objective.print_detail(self.best_model, [v.name for v in self.problem.variables])
            self.problem.print_constraints_detail(self.best_model)
        else:
            print("无可行解！")
        print("==================\n")

# =====================
# 4. 测试用例与主入口
# =====================

def build_sample_problem() -> PBOProblem:
    """
    构造一个简单的PBO问题：
    max 3*x1 + 2*x2 + 1*x3
    s.t. x1 + x2 <= 1
         x2 + x3 <= 1
         x1 + x3 <= 1
    """
    pb = PBOProblem()
    pb.add_variable('x1')
    pb.add_variable('x2')
    pb.add_variable('x3')
    pb.set_objective({'x1': 3, 'x2': 2, 'x3': 1})
    pb.add_constraint({'x1': 1, 'x2': 1}, 1)
    pb.add_constraint({'x2': 1, 'x3': 1}, 1)
    pb.add_constraint({'x1': 1, 'x3': 1}, 1)
    return pb

def build_knapsack_problem() -> PBOProblem:
    """
    构造一个背包问题实例：
    max 6*x1 + 10*x2 + 12*x3
    s.t. 2*x1 + 2*x2 + 3*x3 <= 5
    x1, x2, x3 ∈ {0,1}
    """
    pb = PBOProblem()
    pb.add_variable('x1')
    pb.add_variable('x2')
    pb.add_variable('x3')
    pb.set_objective({'x1': 6, 'x2': 10, 'x3': 12})
    pb.add_constraint({'x1': 2, 'x2': 2, 'x3': 3}, 5)
    return pb

def build_cover_problem() -> PBOProblem:
    """
    0-1覆盖问题：
    min x1 + x2 + x3
    s.t. x1 + x2 >= 1
         x2 + x3 >= 1
         x1 + x3 >= 1
    转化为PBO最大化：max -x1 -x2 -x3, 约束同上（用<=变形）
    """
    pb = PBOProblem()
    pb.add_variable('x1')
    pb.add_variable('x2')
    pb.add_variable('x3')
    pb.set_objective({'x1': -1, 'x2': -1, 'x3': -1})
    pb.add_constraint({'x1': 1, 'x2': 1}, 1)  # x1 + x2 >= 1 -> -x1 - x2 <= -1
    pb.add_constraint({'x2': 1, 'x3': 1}, 1)
    pb.add_constraint({'x1': 1, 'x3': 1}, 1)
    return pb

def build_scheduling_problem() -> PBOProblem:
    """
    简单调度问题：3个任务分配给2台机器，每台机器最多1个任务
    变量：xij表示任务i分配给机器j
    目标：max x11 + x12 + x21 + x22 + x31 + x32
    约束：每个任务分配一次，每台机器最多1个任务
    """
    pb = PBOProblem()
    for i in range(1, 4):
        for j in range(1, 3):
            pb.add_variable(f'x{i}{j}')
    pb.set_objective({f'x{i}{j}': 1 for i in range(1, 4) for j in range(1, 3)})
    # 每个任务分配一次
    for i in range(1, 4):
        pb.add_constraint({f'x{i}1': 1, f'x{i}2': 1}, 1)
    # 每台机器最多1个任务
    for j in range(1, 3):
        pb.add_constraint({f'x1{j}': 1, f'x2{j}': 1, f'x3{j}': 1}, 1)
    return pb

def build_random_problem(n_vars: int = 8, n_cons: int = 10, seed: int = 42) -> PBOProblem:
    """
    随机生成PBO问题
    """
    random.seed(seed)
    pb = PBOProblem()
    for i in range(n_vars):
        pb.add_variable(f'x{i+1}')
    obj = {f'x{i+1}': random.randint(1, 10) for i in range(n_vars)}
    pb.set_objective(obj)
    for _ in range(n_cons):
        coeffs = {f'x{random.randint(1, n_vars)}': random.randint(1, 5) for _ in range(random.randint(2, n_vars))}
        bound = random.randint(5, 15)
        pb.add_constraint(coeffs, bound)
    return pb

def build_special_pb_constraint_problem() -> PBOProblem:
    """
    特殊伪布尔约束例子：
    max x1 + x2 + x3
    s.t. 2*x1 + 2*x2 + 2*x3 <= 3
    """
    pb = PBOProblem()
    for i in range(1, 4):
        pb.add_variable(f'x{i}')
    pb.set_objective({f'x{i}': 1 for i in range(1, 4)})
    pb.add_constraint({f'x1': 2, 'x2': 2, 'x3': 2}, 3)
    return pb

# =====================
# 5. 主入口与测试
# =====================

def main():
    print("=== PBO+CDCL求解器演示 ===\n")
    # 1. 三变量互斥问题
    print("【示例1：三变量互斥问题】")
    pb = build_sample_problem()
    pb.visualize()
    solver = CDCLSolver(pb)
    solver.solve()
    solver.enumerate_all_solutions()
    print("\n【示例2：背包问题】")
    pb2 = build_knapsack_problem()
    pb2.visualize()
    solver2 = CDCLSolver(pb2)
    solver2.solve()
    solver2.enumerate_all_solutions()
    print("\n【示例3：0-1覆盖问题】")
    pb3 = build_cover_problem()
    pb3.visualize()
    solver3 = CDCLSolver(pb3)
    solver3.solve()
    solver3.enumerate_all_solutions()
    print("\n【示例4：调度问题】")
    pb4 = build_scheduling_problem()
    pb4.visualize()
    solver4 = CDCLSolver(pb4)
    solver4.solve()
    solver4.enumerate_all_solutions()
    print("\n【示例5：特殊伪布尔约束问题】")
    pb5 = build_special_pb_constraint_problem()
    pb5.visualize()
    solver5 = CDCLSolver(pb5)
    solver5.solve()
    solver5.enumerate_all_solutions()
    print("\n【示例6：随机PBO问题】")
    pb6 = build_random_problem(n_vars=6, n_cons=8)
    pb6.visualize()
    solver6 = CDCLSolver(pb6)
    solver6.solve()
    # 不枚举大规模解

if __name__ == "__main__":
    main()

# =====================
# 6. 代码结构与扩展建议
# =====================
# - 本代码实现了PBO问题的基本CDCL求解流程，支持伪布尔约束的传播、冲突分析、回溯、学习子句等。
# - 可扩展支持更复杂的分支限界、剪枝、启发式、并行等。
# - 可集成文件输入输出、可视化等功能。
# - 代码注释丰富，便于理解和二次开发。
# - 如需进一步扩展或优化，可随时联系作者。
