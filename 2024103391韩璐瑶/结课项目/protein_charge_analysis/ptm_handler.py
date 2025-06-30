import csv

class PTMHandler:
    def __init__(self, ptm_config):
        self.ptm_config = ptm_config or []
    
    def get_ptm_effect(self, amino_acid, position, pH):
        """获取PTM对电荷的影响"""
        effect = 0.0
        for ptm in self.ptm_config:
            # 检查位置和氨基酸类型是否匹配
            if (ptm['position'] == position and 
                (ptm['amino_acid'] == amino_acid or ptm['amino_acid'] == 'ANY')):
                
                # 直接电荷效应
                direct_effect = float(ptm.get('effect', 0))
                
                # pKa依赖效应
                pKa_effect = 0
                if 'pKa' in ptm and 'pKa_effect' in ptm:
                    pKa = float(ptm['pKa'])
                    pKa_effect_val = float(ptm['pKa_effect'])
                    
                    if ptm['effect_type'] == 'acidic':
                        pKa_effect = -pKa_effect_val / (1 + 10**(pKa - pH))
                    elif ptm['effect_type'] == 'basic':
                        pKa_effect = pKa_effect_val / (1 + 10**(pH - pKa))
                
                effect += direct_effect + pKa_effect
                
        return effect
    
    def get_ptm_summary(self):
        """获取PTM配置摘要"""
        return self.ptm_config