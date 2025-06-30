import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import math

class ProteinChargeAnalyzer:
    # 氨基酸解离常数 (完整版)
    DEFAULT_AMINO_ACID_pKa = {
        # 酸性残基 (带负电)
        'D': {'pKa': 3.90, 'type': 'acidic', 'weight': 133.10},  # Aspartic acid
        'E': {'pKa': 4.07, 'type': 'acidic', 'weight': 147.13},  # Glutamic acid
        'C': {'pKa': 8.18, 'type': 'acidic', 'weight': 121.16},  # Cysteine
        'Y': {'pKa': 10.46, 'type': 'acidic', 'weight': 181.19}, # Tyrosine
        # 碱性残基 (带正电)
        'H': {'pKa': 6.04, 'type': 'basic', 'weight': 155.16},   # Histidine
        'K': {'pKa': 10.54, 'type': 'basic', 'weight': 146.19},  # Lysine
        'R': {'pKa': 12.48, 'type': 'basic', 'weight': 174.20},  # Arginine
        # 末端基团
        'N_term': {'pKa': 9.60, 'type': 'basic', 'weight': 0},  # N-terminal
        'C_term': {'pKa': 2.34, 'type': 'acidic', 'weight': 0}  # C-terminal
    }
    
    def __init__(self, sequence, custom_pka=None, ptm_handler=None, 
                 temperature=25.0, ionic_strength=0.15):
        self.sequence = sequence.upper()
        self.charge_profile = []
        self.temperature_effect = []
        self.pI = None
        self.amino_acid_composition = None
        self.ptm_handler = ptm_handler
        self.temperature = temperature  # 温度(℃)
        self.ionic_strength = ionic_strength  # 离子强度(M)
        
        # 使用自定义pKa值或默认值
        self.pka_table = custom_pka if custom_pka else self.DEFAULT_AMINO_ACID_pKa
        
        # 调整pKa值以适应温度
        self._adjust_pka_for_temperature()
        
        # 分析氨基酸组成
        self._analyze_amino_acid_composition()
        
        # 计算分子量
        self.molecular_weight = self._calculate_molecular_weight()
    
    def _adjust_pka_for_temperature(self):
        """根据温度调整pKa值 (每升高1℃, pKa降低约0.01)"""
        temp_adjustment = (self.temperature - 25.0) * 0.01
        for aa, data in self.pka_table.items():
            data['pKa'] = round(data['pKa'] - temp_adjustment, 4)
    
    def _calculate_molecular_weight(self):
        """计算蛋白质分子量"""
        water_weight = 18.02  # 水的分子量
        total_weight = 0
        
        # 计算所有氨基酸的分子量总和
        for aa in self.sequence:
            if aa in self.pka_table and 'weight' in self.pka_table[aa]:
                total_weight += self.pka_table[aa]['weight']
        
        # 减去肽键形成时失去的水分子 (n-1个水分子)
        n_residues = len(self.sequence)
        total_weight -= (n_residues - 1) * water_weight
        
        # 加上末端基团
        if 'N_term' in self.pka_table and 'weight' in self.pka_table['N_term']:
            total_weight += self.pka_table['N_term']['weight']
        if 'C_term' in self.pka_table and 'weight' in self.pka_table['C_term']:
            total_weight += self.pka_table['C_term']['weight']
        
        return total_weight
    
    def _analyze_amino_acid_composition(self):
        """分析氨基酸组成"""
        counter = Counter(self.sequence)
        total = len(self.sequence)
        self.amino_acid_composition = {
            aa: {
                'count': count, 
                'percentage': count/total*100,
                'type': self.pka_table.get(aa, {}).get('type', 'neutral')
            }
            for aa, count in counter.items()
        }
    
    def _calculate_residue_charge(self, aa, pH, position=None):
        """计算单个残基在给定pH下的电荷"""
        if aa not in self.pka_table:
            return 0
            
        data = self.pka_table[aa]
        pKa = data['pKa']
        charge = 0
        
        # 考虑离子强度影响 (Debye-Huckel近似)
        ionic_factor = 1 - 0.5 * math.sqrt(self.ionic_strength)
        
        if data['type'] == 'acidic':
            # 酸性基团：charge = -1 / (1 + 10^(pKa - pH))
            charge = -1 / (1 + 10**(pKa - pH)) * ionic_factor
        elif data['type'] == 'basic':
            # 碱性基团：charge = 1 / (1 + 10^(pH - pKa))
            charge = 1 / (1 + 10**(pH - pKa)) * ionic_factor
        
        # 应用翻译后修饰
        if self.ptm_handler and position is not None:
            charge += self.ptm_handler.get_ptm_effect(aa, position, pH)
            
        return charge

    def calculate_net_charge(self, pH):
        """计算特定pH下的净电荷"""
        charge = 0.0
        
        # N末端和C末端
        charge += self._calculate_residue_charge('N_term', pH, position=1)
        charge += self._calculate_residue_charge('C_term', pH, position=len(self.sequence))
        
        # 氨基酸残基
        for idx, aa in enumerate(self.sequence):
            position = idx + 1
            charge += self._calculate_residue_charge(aa, pH, position)
            
        return charge

    def find_pI(self, precision=0.001, max_iter=100):
        """使用二分法计算等电点"""
        low, high = 0.0, 14.0
        iter_count = 0
        
        # 检查初始边界
        if self.calculate_net_charge(low) < 0:
            return low
        if self.calculate_net_charge(high) > 0:
            return high
        
        while high - low > precision and iter_count < max_iter:
            mid = (low + high) / 2
            charge = self.calculate_net_charge(mid)
            
            if abs(charge) < precision:
                break
                
            if charge > 0:
                low = mid
            else:
                high = mid
                
            iter_count += 1
            
        self.pI = round((low + high) / 2, 4)
        return self.pI

    def generate_charge_profile(self, pH_range=np.arange(0, 14.1, 0.1)):
        """生成电荷分布曲线"""
        self.charge_profile = [
            (pH, self.calculate_net_charge(pH)) 
            for pH in pH_range
        ]
        return self.charge_profile

    def analyze_temperature_effect(self, temp_range=np.arange(0, 101, 5)):
        """分析温度对等电点的影响"""
        original_temp = self.temperature
        results = []
        
        for temp in temp_range:
            self.temperature = temp
            self._adjust_pka_for_temperature()
            pI = self.find_pI()
            results.append((temp, pI))
        
        # 恢复原始温度
        self.temperature = original_temp
        self._adjust_pka_for_temperature()
        self.temperature_effect = results
        return results

    def predict_electrophoresis(self, pH):
        """预测电泳迁移行为"""
        charge = self.calculate_net_charge(pH)
        direction = "向阳极" if charge < 0 else "向阴极" if charge > 0 else "不移动"
        
        # 迁移率计算 (简化模型)
        mobility = abs(charge) * 100 / (len(self.sequence) ** 0.5)
        mobility = min(100, max(0, mobility))  # 限制在0-100%范围内
        
        return direction, mobility

    def plot_charge_distribution(self, save_path=None):
        """绘制电荷-pH曲线"""
        if not self.charge_profile:
            self.generate_charge_profile()
            
        pH, charge = zip(*self.charge_profile)
        plt.figure(figsize=(10, 6))
        plt.plot(pH, charge, 'b-', linewidth=2)
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(self.pI, color='r', linestyle=':', 
                   label=f'pI = {self.pI}')
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('净电荷', fontsize=12)
        plt.title(f'蛋白质电荷随pH变化曲线\n温度={self.temperature}℃, 离子强度={self.ionic_strength}M', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

    def plot_temperature_effect(self, save_path=None):
        """绘制温度对pI的影响"""
        if not self.temperature_effect:
            self.analyze_temperature_effect()
            
        temp, pI = zip(*self.temperature_effect)
        plt.figure(figsize=(10, 6))
        plt.plot(temp, pI, 'r-o', linewidth=2, markersize=6)
        plt.xlabel('温度 (℃)', fontsize=12)
        plt.ylabel('等电点 (pI)', fontsize=12)
        plt.title('温度对等电点的影响', fontsize=14)
        plt.grid(alpha=0.2)
        
        # 添加回归线
        if len(temp) > 1:
            z = np.polyfit(temp, pI, 1)
            p = np.poly1d(z)
            plt.plot(temp, p(temp), "b--", label=f"趋势线: pI = {z[0]:.4f}*T + {z[1]:.2f}")
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_aa_composition(self, save_path=None):
        """绘制氨基酸组成图"""
        aa_data = self.amino_acid_composition
        if not aa_data:
            return
            
        # 按类型分组
        aa_types = {'acidic': [], 'basic': [], 'neutral': []}
        labels = []
        sizes = []
        colors = []
        
        for aa, data in aa_data.items():
            labels.append(aa)
            sizes.append(data['percentage'])
            if data['type'] == 'acidic':
                colors.append('red')
                aa_types['acidic'].append(aa)
            elif data['type'] == 'basic':
                colors.append('blue')
                aa_types['basic'].append(aa)
            else:
                colors.append('gray')
                aa_types['neutral'].append(aa)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 饼图
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True)
        ax1.axis('equal')
        ax1.set_title('氨基酸组成分布')
        
        # 条形图 - 修复了列表推导式错误
        types = list(aa_types.keys())
        
        # 正确计算每个类型的总百分比
        percentages = [
            sum(aa_data[aa]['percentage'] for aa in aa_types['acidic']),
            sum(aa_data[aa]['percentage'] for aa in aa_types['basic']),
            sum(aa_data[aa]['percentage'] for aa in aa_types['neutral'])
        ]
        
        # 绘制条形图
        ax2.bar(types, percentages, color=['red', 'blue', 'gray'])
        ax2.set_xlabel('氨基酸类型')
        ax2.set_ylabel('百分比 (%)')
        ax2.set_title('按电荷类型分组')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def get_amino_acid_charge_at_ph(self, pH):
        """获取每个氨基酸在指定pH下的电荷"""
        charge_dict = {}
        for aa in set(self.sequence):
            charge_dict[aa] = self._calculate_residue_charge(aa, pH)
        
        # 添加末端电荷
        charge_dict['N_term'] = self._calculate_residue_charge('N_term', pH, position=1)
        charge_dict['C_term'] = self._calculate_residue_charge('C_term', pH, position=len(self.sequence))
        return charge_dict
    
    def get_amino_acid_composition(self):
        """获取氨基酸组成"""
        return self.amino_acid_composition
    
    def get_charge_contributors(self, pH):
        """获取主要电荷贡献者"""
        contributors = {}
        
        # N末端和C末端
        contributors['N_term'] = self._calculate_residue_charge('N_term', pH, position=1)
        contributors['C_term'] = self._calculate_residue_charge('C_term', pH, position=len(self.sequence))
        
        # 按氨基酸类型分组
        for aa in set(self.sequence):
            charge = self._calculate_residue_charge(aa, pH)
            if abs(charge) > 0.01:  # 只考虑有显著贡献的
                count = self.amino_acid_composition[aa]['count']
                total_charge = charge * count
                contributors[aa] = total_charge
                
        return dict(sorted(contributors.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def get_physical_properties(self):
        """获取物理性质摘要"""
        return {
            'length': len(self.sequence),
            'molecular_weight': self.molecular_weight,
            'pI': self.pI,
            'temperature': self.temperature,
            'ionic_strength': self.ionic_strength
        }