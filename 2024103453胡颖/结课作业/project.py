import math
import collections
import statistics

STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL"  # 硒代半胱氨酸和吡咯赖氨酸有时也被认为是标准氨基酸，这里也包含
}

# 结构体用于存储解析后的原子数据
class Atom:
    """表示PDB文件中的一个原子。"""
    def __init__(self, record_type, 
                 atom_serial_number, 
                 atom_name, alt_loc,
                 residue_name, 
                 chain_id, 
                 residue_sequence_number, 
                 insertion_code,
                 x, 
                 y, 
                 z, 
                 occupancy, 
                 b_factor, 
                 segment_id, 
                 element_symbol, 
                 charge):
        self.record_type = record_type
        self.atom_serial_number = int(atom_serial_number)
        self.atom_name = atom_name.strip()
        self.alt_loc = alt_loc.strip()
        self.residue_name = residue_name.strip()
        self.chain_id = chain_id.strip()
        self.residue_sequence_number = int(residue_sequence_number)
        self.insertion_code = insertion_code.strip()
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.occupancy = float(occupancy)
        self.b_factor = float(b_factor)
        self.segment_id = segment_id.strip()
        self.element_symbol = element_symbol.strip()
        self.charge = charge.strip()

    def __repr__(self):
        return (f"Atom(name='{self.atom_name}', 
                res='{self.residue_name}', 
                "f"res_seq={self.residue_sequence_number}, 
                chain='{self.chain_id}', 
                "f"coords=({self.x:.2f},{self.y:.2f},{self.z:.2f}))")

# 结构体用于存储解析后的残基数据
class Residue:
    """表示PDB文件中的一个残基，包含其所有原子。"""
    def __init__(self, residue_name, chain_id, residue_sequence_number, insertion_code):
        self.residue_name = residue_name
        self.chain_id = chain_id
        self.residue_sequence_number = residue_sequence_number
        self.insertion_code = insertion_code
        self.atoms = []
        self.is_standard = residue_name in STANDARD_AMINO_ACIDS

    def add_atom(self, atom):
        """向残基中添加一个原子。"""
        self.atoms.append(atom)

    def get_atom(self, atom_name):
        """根据原子名称获取残基中的原子。"""
        for atom in self.atoms:
            if atom.atom_name == atom_name:
                return atom
        return None

    def __repr__(self):
        return (f"Residue(name='{self.residue_name}', chain='{self.chain_id}', "
                f"seq={self.residue_sequence_number}, atoms={len(self.atoms)})")

# 主解析函数
def parse_pdb_file(pdb_file_path):
    """
    解析PDB文件，提取原子信息、残基信息和分辨率。
    返回一个包含所有原子、残基和分辨率的字典。
    """
    atoms = []
    residues_dict = collections.defaultdict(lambda: collections.defaultdict(Residue))
    resolution = None
    
    try:
        with open(pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    record_type = line[0:6].strip()
                    atom_serial_number = line[6:11].strip()
                    atom_name = line[12:16].strip()
                    alt_loc = line[16:17].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_sequence_number = line[22:26].strip()
                    insertion_code = line[26:27].strip()
                    x = line[30:38].strip()
                    y = line[38:46].strip()
                    z = line[46:54].strip()
                    occupancy = line[54:60].strip()
                    b_factor = line[60:66].strip()
                    segment_id = line[72:76].strip()
                    element_symbol = line[76:78].strip()
                    charge = line[78:80].strip()

                    atom = Atom(record_type, atom_serial_number, atom_name, alt_loc,
                                residue_name, chain_id, residue_sequence_number,
                                insertion_code, x, y, z, occupancy, b_factor,
                                segment_id, element_symbol, charge)
                    atoms.append(atom)

                    # 将原子添加到其对应的残基中
                    residue_key = (atom.chain_id, atom.residue_sequence_number, atom.insertion_code)
                    if residue_key not in residues_dict[atom.chain_id]:
                         residues_dict[atom.chain_id][residue_key] = Residue(
                             atom.residue_name, atom.chain_id, atom.residue_sequence_number, atom.insertion_code
                         )
                    residues_dict[atom.chain_id][residue_key].add_atom(atom)

                elif line.startswith("REMARK   2 RESOLUTION."):
                    # 尝试从REMARK 2中提取分辨率
                    parts = line.split()
                    if "RESOLUTION." in parts and "ANGSTROMS." in parts:
                        try:
                            # 查找 "RESOLUTION." 和 "ANGSTROMS." 之间的数字
                            for i, part in enumerate(parts):
                                if part == "RESOLUTION." and i + 1 < len(parts):
                                    # 尝试转换下一个单词为浮点数
                                    if i + 2 < len(parts) and parts[i+2] == "ANGSTROMS.":
                                        resolution = float(parts[i+1])
                                        break
                                elif part == "ANGSTROMS." and i > 0:
                                    try:
                                        resolution = float(parts[i-1])
                                        break
                                    except ValueError:
                                        continue
                        except ValueError:
                            resolution = None # 解析失败
                elif line.startswith("REMARK   3  RESOLUTION."):
                    # 尝试从REMARK 3中提取分辨率，有时格式不同
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith("ANGSTROMS.") and i > 0:
                            try:
                                resolution = float(parts[i-1])
                                break
                            except ValueError:
                                continue
    except FileNotFoundError:
      
        print(f"错误: 文件 '{pdb_file_path}' 未找到。")
        return None
    except Exception as e:
      
        print(f"解析PDB文件时发生错误: {e}")
        return None

    # 将嵌套的defaultdict转换为更扁平的残基列表，并按链和序列号排序
    all_residues = []
    for chain_id in sorted(residues_dict.keys()):
        sorted_residues_in_chain = sorted(
            residues_dict[chain_id].values(),
            key=lambda r: (r.residue_sequence_number, r.insertion_code)
        )
        all_residues.extend(sorted_residues_in_chain)

    return {
        "atoms": atoms,
        "residues": all_residues,
        "resolution": resolution
    }

def euclidean_distance(atom1, atom2):
  
    """计算两个原子之间的欧几里得距离。"""
    return math.sqrt((atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2 + (atom1.z - atom2.z)**2)

# --- 低质量结构检测标准实现 ---

def check_non_standard_amino_acid_percentage(residues):
    """
    1. 非标准氨基酸个数占比 > 50%
    计算非标准氨基酸残基的百分比。
    如果超过50%，则标记为低质量。
    """
    if not residues:
        return False, "没有残基数据可供检查。"

    total_amino_acids = 0
    non_standard_amino_acids = 0

    for res in residues:
        # 只考虑蛋白质链中的残基，排除HETATM中的小分子
        # 简单判断方法：如果残基是标准氨基酸或非标准但通常与蛋白质相关的氨基酸，则计数
        # 这里我们只检查Residue对象中的is_standard标志
        # PDB文件中HETATM记录通常用于配体、水分子等，ATOM记录用于蛋白质/核酸
        # 我们的解析器将所有记录都解析为Atom对象，但Residue对象只基于 residue_name 进行标准/非标准判断
        # 实际应用中，可能需要根据 record_type (ATOM/HETATM) 进一步筛选

        total_amino_acids += 1
        if not res.is_standard:
            non_standard_amino_acids += 1

    if total_amino_acids == 0:
        return False, "未发现蛋白质残基。"

    percentage = (non_standard_amino_acids / total_amino_acids) * 100
    
    if percentage > 50.0:
      
        return True, (f"非标准氨基酸占比过高: {percentage:.2f}% (非标准: {non_standard_amino_acids}, "
                      f"总数: {total_amino_acids})，超过50%。")
      
    return False, (f"非标准氨基酸占比: {percentage:.2f}% (非标准: {non_standard_amino_acids}, "
                   f"总数: {total_amino_acids})，未超过50%。")

def check_missing_backbone_atoms(residues, threshold_percentage=10.0):
    """
    2. 缺失主链原子 (N, CA, C, O) 占比过高
    检查每个残基是否含有N、CA、C、O原子。
    如果超过阈值百分比的残基缺失这些原子，则标记为低质量。
    """
    if not residues:
      
        return False, "没有残基数据可供检查。"

    missing_backbone_residues_count = 0
    total_residues_with_backbone_check = 0

    # 针对蛋白质残基进行检查，跳过可能不是氨基酸的HETATM残基
    for res in residues:
      
        # 假设我们只关心蛋白质残基的主链完整性，并且这些残基是标准或非标准氨基酸
        # 进一步过滤 HETATM 类型残基，如果 Residue 对象有相应的标志 (目前没有)
        # 或者我们只检查 is_standard 的残基
        if not res.is_standard and res.residue_name not in STANDARD_AMINO_ACIDS:
            # 如果是非标准且不在我们定义的标准列表，可能不是蛋白质残基，跳过
            continue
        
        # 检查 N, CA, C, O 原子是否存在
        n_atom = res.get_atom('N')
        ca_atom = res.get_atom('CA')
        c_atom = res.get_atom('C')
        o_atom = res.get_atom('O')

        has_n = n_atom is not None
        has_ca = ca_atom is not None
        has_c = c_atom is not None
        has_o = o_atom is not None

        if not (has_n and has_ca and has_c and has_o):
            missing_backbone_residues_count += 1
        total_residues_with_backbone_check += 1

    if total_residues_with_backbone_check == 0:
      
        return False, "未发现可检查主链原子的蛋白质残基。"

    percentage_missing = (missing_backbone_residues_count / total_residues_with_backbone_check) * 100

    if percentage_missing > threshold_percentage:
      
        return True, (f"缺失主链原子残基占比过高: {percentage_missing:.2f}% (缺失: {missing_backbone_residues_count}, "
                      f"总数: {total_residues_with_backbone_check})，超过{threshold_percentage:.1f}%。")
    return False, (f"缺失主链原子残基占比: {percentage_missing:.2f}% (缺失: {missing_backbone_residues_count}, "
                   f"总数: {total_residues_with_backbone_check})，未超过{threshold_percentage:.1f}%。")

def check_b_factor_deviation(atoms, b_factor_std_dev_threshold=20.0, avg_b_factor_threshold=50.0):
    """
    3. B因子/温度因子过高或标准差过大
    检查所有原子的B因子，如果平均B因子过高或B因子的标准差过大，则标记为低质量。
    """
    if not atoms:
        return False, "没有原子数据可供检查。"

    b_factors = [atom.b_factor for atom in atoms]
    
    if len(b_factors) < 2: # 需要至少两个B因子才能计算标准差
        avg_b_factor = sum(b_factors) / len(b_factors) if b_factors else 0
        if avg_b_factor > avg_b_factor_threshold:
            return True, f"B因子数据不足，但平均B因子过高: {avg_b_factor:.2f}，超过{avg_b_factor_threshold:.1f}。"
        return False, f"B因子数据不足，平均B因子: {avg_b_factor:.2f}。"

    avg_b_factor = statistics.mean(b_factors)
    std_dev_b_factor = statistics.stdev(b_factors)

    reasons = []
    if avg_b_factor > avg_b_factor_threshold:
        reasons.append(f"平均B因子过高 ({avg_b_factor:.2f} > {avg_b_factor_threshold:.1f})")
    if std_dev_b_factor > b_factor_std_dev_threshold:
        reasons.append(f"B因子标准差过大 ({std_dev_b_factor:.2f} > {b_factor_std_dev_threshold:.1f})")

    if reasons:
        return True, f"B因子异常: " + ", ".join(reasons) + "."
    return False, (f"B因子正常 (平均: {avg_b_factor:.2f}, 标准差: {std_dev_b_factor:.2f}，"
                   f"均未超过阈值 {avg_b_factor_threshold:.1f}/{b_factor_std_dev_threshold:.1f})。")

def check_atom_clashes(atoms, clash_threshold=1.5):
    """
    4. 原子间严重碰撞 (clashes)
    检查所有非键合原子对之间的距离。如果任何一对非键合原子距离小于`clash_threshold`，则标记为低质量。
    注意：此操作对于大型结构可能计算量巨大 (O(N^2))。
    """
    if not atoms or len(atoms) < 2:
      
        return False, "没有足够的原子数据可供检查碰撞。"

    clashes_found = []
    # 简单的N^2检查，仅用于演示。生产环境应使用空间索引 (如KD-Tree)。
    for i in range(len(atoms)):
      
        for j in range(i + 1, len(atoms)):
          
            atom1 = atoms[i]
            atom2 = atoms[j]

            # 排除同一残基内的原子或相邻残基的键合原子（简化处理）
            # 更精确的碰撞检查需要键合信息。这里只是一个粗略的非键合原子检查。
            # 如果两个原子属于同一个残基，并且它们的残基序列号和链ID都相同，则跳过
            if (atom1.chain_id == atom2.chain_id and
                
                atom1.residue_sequence_number == atom2.residue_sequence_number and
                atom1.insertion_code == atom2.insertion_code):
                continue # 同一残基的原子，假定不会发生内部碰撞 (由键合规则控制)

            distance = euclidean_distance(atom1, atom2)
            if distance < clash_threshold:
              
                clashes_found.append(f"严重碰撞: {atom1.atom_name}{atom1.residue_sequence_number}{atom1.chain_id} - "
                                     f"{atom2.atom_name}{atom2.residue_sequence_number}{atom2.chain_id} (距离: {distance:.2f} Å)")
                # 找到一个碰撞就足够，可以提前返回
                if len(clashes_found) > 0: # 避免记录太多，只返回第一个
                    return True, f"发现严重原子碰撞: {clashes_found[0]}。"

    if clashes_found:
        return True, f"发现{len(clashes_found)}个原子碰撞，例如: {clashes_found[0]}"
    return False, "未发现严重原子碰撞。"

def check_chain_breaks(residues, c_n_distance_threshold=2.0):
    """
    5. 异常的肽链连接 (chain breaks)
    检查连续残基C原子和N原子之间的距离。如果距离过大，则可能存在肽链断裂。
    标准肽键C-N距离约为1.32Å。
    """
    if not residues or len(residues) < 2:
        return False, "没有足够的残基可供检查链断裂。"

    chain_breaks_found = []
    
    # 按链和残基序列号排序的残基列表
    # parse_pdb_file 已经返回了排序后的残基列表

    for i in range(len(residues) - 1):
        res1 = residues[i]
        res2 = residues[i+1]

        # 检查是否是同一条链上的连续残基
        if res1.chain_id == res2.chain_id and \
           (res2.residue_sequence_number == res1.residue_sequence_number + 1 or
            (res2.residue_sequence_number == res1.residue_sequence_number and
             # 处理插入码的情况，例如 10A, 10B 这种
             ord(res2.insertion_code) == ord(res1.insertion_code) + 1 if res1.insertion_code and res2.insertion_code else False
            )):
            
            c_atom_res1 = res1.get_atom('C')
            n_atom_res2 = res2.get_atom('N')

            if c_atom_res1 and n_atom_res2:
                distance = euclidean_distance(c_atom_res1, n_atom_res2)
                if distance > c_n_distance_threshold:
                    chain_breaks_found.append(f"链断裂: {res1.residue_name}{res1.residue_sequence_number}{res1.chain_id} (C原子) - "
                                              f"{res2.residue_name}{res2.residue_sequence_number}{res2.chain_id} (N原子) "
                                              f"(距离: {distance:.2f} Å, 期望 < {c_n_distance_threshold:.1f} Å)")
        # 如果不是连续的残基，但链ID相同，则也认为可能存在断裂
        elif res1.chain_id == res2.chain_id:
            # 这种情况可能是缺失了中间的残基
            chain_breaks_found.append(f"潜在链断裂/缺失残基: {res1.residue_name}{res1.residue_sequence_number}{res1.chain_id} - "
                                      f"{res2.residue_name}{res2.residue_sequence_number}{res2.chain_id} (不连续序列号)")


    if chain_breaks_found:
        return True, f"发现{len(chain_breaks_found)}处肽链断裂或不连续，例如: {chain_breaks_found[0]}"
    return False, "未发现异常的肽链连接或断裂。"

def check_resolution(resolution, resolution_threshold=3.0):
    """
    6. 分辨率过低
    如果PDB文件头中包含分辨率信息，且分辨率高于给定阈值（例如3.0 Å），则标记为低质量。
    """
    if resolution is None:
        return False, "PDB文件中未找到分辨率信息。"
    
    if resolution > resolution_threshold:
        return True, f"分辨率过低: {resolution:.2f} Å，超过{resolution_threshold:.1f} Å。"
    return False, f"分辨率正常: {resolution:.2f} Å，未超过{resolution_threshold:.1f} Å。"


def detect_low_quality_structure(pdb_file_path):
    """
    主函数：检测蛋白质结构质量。
    对所有定义的标准进行检查，并汇总结果。
    """
    print(f"--- 正在检测文件: {pdb_file_path} ---")
    parsed_data = parse_pdb_file(pdb_file_path)

    if parsed_data is None:
        return {"status": "失败", "reason": "PDB文件解析失败或文件不存在。"}

    atoms = parsed_data["atoms"]
    residues = parsed_data["residues"]
    resolution = parsed_data["resolution"]

    quality_issues = []
    
    # 1. 非标准氨基酸个数占比 > 50%
    is_low_quality_1, reason_1 = check_non_standard_amino_acid_percentage(residues)
  
    if is_low_quality_1:
        quality_issues.append(f"问题1: {reason_1}")
    print(f"检查1 (非标准氨基酸占比): {reason_1}")

    # 2. 缺失主链原子 (N, CA, C, O) 占比过高
    is_low_quality_2, reason_2 = check_missing_backbone_atoms(residues)
  
    if is_low_quality_2:
        quality_issues.append(f"问题2: {reason_2}")
    print(f"检查2 (缺失主链原子): {reason_2}")

    # 3. B因子/温度因子过高或标准差过大
    is_low_quality_3, reason_3 = check_b_factor_deviation(atoms)
  
    if is_low_quality_3:
        quality_issues.append(f"问题3: {reason_3}")
    print(f"检查3 (B因子异常): {reason_3}")

    # 4. 原子间严重碰撞 (clashes)
    is_low_quality_4, reason_4 = check_atom_clashes(atoms)
  
    if is_low_quality_4:
        quality_issues.append(f"问题4: {reason_4}")
    print(f"检查4 (原子碰撞): {reason_4}")

    # 5. 异常的肽链连接 (chain breaks)
    is_low_quality_5, reason_5 = check_chain_breaks(residues)
  
    if is_low_quality_5:
      
        quality_issues.append(f"问题5: {reason_5}")
      
    print(f"检查5 (链断裂): {reason_5}")

    # 6. 分辨率过低
    is_low_quality_6, reason_6 = check_resolution(resolution)
  
    if is_low_quality_6:
        quality_issues.append(f"问题6: {reason_6}")
    print(f"检查6 (分辨率): {reason_6}")

    if quality_issues:
        return {
            "status": "低质量",
            "issues": quality_issues,
            "summary": f"该蛋白质结构被判定为低质量，共发现 {len(quality_issues)} 个问题。"
        }
    else:
        return {
            "status": "高质量",
            "issues": [],
            "summary": "该蛋白质结构看起来质量良好。"
        }

if __name__ == "__main__":

    test_pdb_content_low_quality = "请输入.pdb文件"
    test_pdb_content_high_quality = "请输入.pdb文件"
    
    # 将测试内容写入临时文件
    low_quality_pdb_file = "low_quality_test.pdb"
    with open(low_quality_pdb_file, "w") as f:
        f.write(test_pdb_content_low_quality)

    high_quality_pdb_file = "high_quality_test.pdb"
    with open(high_quality_pdb_file, "w") as f:
        f.write(test_pdb_content_high_quality)

    # 运行低质量测试
    print("\n--- 运行低质量PDB文件测试 ---")
    result_low_quality = detect_low_quality_structure(low_quality_pdb_file)
  
    print("\n检测结果 (低质量):")
    print(result_low_quality)

    print("\n" + "="*50 + "\n")

    # 运行高质量测试
    print("--- 运行高质量PDB文件测试 ---")
    result_high_quality = detect_low_quality_structure(high_quality_pdb_file)
  
    print("\n检测结果 (高质量):")
    print(result_high_quality)
