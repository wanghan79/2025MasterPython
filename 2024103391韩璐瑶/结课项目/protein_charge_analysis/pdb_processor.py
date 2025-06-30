import os
import numpy as np
from Bio.PDB import PDBParser, DSSP, PPBuilder

class PDBProcessor:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.structure = None
        self.parser = PDBParser()
        self.load_structure()
    
    def load_structure(self):
        """加载PDB结构"""
        try:
            self.structure = self.parser.get_structure('protein', self.pdb_file)
        except Exception as e:
            raise RuntimeError(f"加载PDB文件失败: {e}")
    
    def get_sequence(self, chain_id='A'):
        """获取指定链的氨基酸序列"""
        sequence = ""
        for model in self.structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        # 检查是否是氨基酸残基
                        if residue.id[0] == ' ':  # 排除水分子、离子等
                            try:
                                residue_name = residue.get_resname()
                                # 将三字母代码转换为单字母代码
                                single_letter = self._aa3to1(residue_name)
                                sequence += single_letter
                            except KeyError:
                                # 处理非标准氨基酸（跳过）
                                continue
        return sequence
    
    def analyze_structure(self):
        """分析PDB结构"""
        results = {
            'chains': [],
            'secondary_structure': {},
            'surface_accessibility': {}
        }
        
        for model in self.structure:
            # 获取链信息
            for chain in model:
                chain_info = {
                    'id': chain.id,
                    'length': len(list(chain.get_residues())),
                    'sequence': self.get_sequence(chain.id)
                }
                results['chains'].append(chain_info)
            
            # 使用DSSP分析二级结构和可及表面积
            try:
                dssp = DSSP(model, self.pdb_file)
                for (chain_id, res_id), values in dssp.property_dict.items():
                    if chain_id not in results['secondary_structure']:
                        results['secondary_structure'][chain_id] = {}
                        results['surface_accessibility'][chain_id] = {}
                    
                    results['secondary_structure'][chain_id][res_id] = values[2]
                    results['surface_accessibility'][chain_id][res_id] = values[3]
            except Exception as e:
                print(f"DSSP分析失败: {e}")
        
        return results
    
    def generate_pymol_script(self, analyzer, pH, output_prefix="protein_charge"):
        """生成PyMOL电荷可视化脚本"""
        # 创建电荷映射
        charge_map = {}
        for idx, aa in enumerate(analyzer.sequence):
            position = idx + 1
            charge = analyzer._calculate_residue_charge(aa, pH, position)
            charge_map[position] = charge
        
        # 创建PyMOL脚本
        script = f"""# PyMOL电荷可视化脚本
# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 蛋白质序列: {analyzer.sequence[:20]}... (共{len(analyzer.sequence)}aa)
# pH = {pH}, pI = {analyzer.pI}

# 加载结构
cmd.load("{os.path.abspath(self.pdb_file)}", "protein")

# 清除现有可视化
cmd.hide("everything")
cmd.show("cartoon", "protein")
cmd.set("cartoon_transparency", 0.2)

# 设置电荷颜色映射
cmd.set_color("neg_charge", [1.0, 0.2, 0.2])  # 红色: 负电荷
cmd.set_color("pos_charge", [0.2, 0.2, 1.0])  # 蓝色: 正电荷
cmd.set_color("neutral", [0.8, 0.8, 0.8])     # 灰色: 中性

# 应用电荷映射
"""
        
        # 为每个残基应用颜色
        for pos, charge in charge_map.items():
            if charge < -0.1:
                color = "neg_charge"
            elif charge > 0.1:
                color = "pos_charge"
            else:
                color = "neutral"
            
            script += f"cmd.color(\"{color}\", \"resi {pos}\")\n"
        
        # 添加图例
        script += """
# 创建图例
cmd.pseudoatom("negative", pos=[0,0,0], label="负电荷")
cmd.pseudoatom("positive", pos=[0,0,0], label="正电荷")
cmd.pseudatom("neutral", pos=[0,0,0], label="中性")

cmd.group("legend", "negative positive neutral")
cmd.hide("everything", "legend")
cmd.label("legend", "label")

# 设置视图
cmd.orient()
cmd.zoom(complete=1)
cmd.ray(800, 600)
cmd.png("{}_charge.png", dpi=300)
""".format(output_prefix)
        
        return script
    
    @staticmethod
    def _aa3to1(residue_name):
        """将三字母氨基酸代码转换为单字母代码"""
        conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O'
        }
        return conversion.get(residue_name, 'X')  # 未知氨基酸用X表示