import os
import base64
import time
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

def generate_html_report(analyzer, electrophoresis_ph, direction, mobility,
                         pdb_analysis=None, output_file='report.html'):
    """
    生成HTML分析报告
    :param analyzer: ProteinChargeAnalyzer实例
    :param electrophoresis_ph: 电泳pH条件
    :param direction: 电泳迁移方向
    :param mobility: 电泳迁移率
    :param pdb_analysis: PDB分析结果
    :param output_file: 输出文件名
    """
    # 创建图像缓冲区
    img_buffers = {}
    
    # 函数：保存图表到缓冲区
    def save_plot_to_buffer(func, **kwargs):
        plt.figure()
        func(**kwargs)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    # 生成图表
    img_buffers['charge_plot'] = save_plot_to_buffer(analyzer.plot_charge_distribution)
    img_buffers['temp_plot'] = save_plot_to_buffer(analyzer.plot_temperature_effect)
    img_buffers['aa_plot'] = save_plot_to_buffer(analyzer.plot_aa_composition)
    
    # 创建氨基酸组成表格
    aa_composition = analyzer.get_amino_acid_composition()
    aa_rows = []
    for aa, data in aa_composition.items():
        aa_rows.append({
            'amino_acid': aa,
            'count': data['count'],
            'percentage': f"{data['percentage']:.2f}%",
            'type': data['type']
        })
    
    # 获取氨基酸电荷分布
    aa_charge = analyzer.get_amino_acid_charge_at_ph(electrophoresis_ph)
    
    # 准备模板数据
    context = {
        'sequence': analyzer.sequence,
        'sequence_length': len(analyzer.sequence),
        'pI': analyzer.pI,
        'molecular_weight': analyzer.molecular_weight,
        'temperature': analyzer.temperature,
        'ionic_strength': analyzer.ionic_strength,
        'electrophoresis_ph': electrophoresis_ph,
        'charge_at_ph': round(analyzer.calculate_net_charge(electrophoresis_ph), 4),
        'direction': direction,
        'mobility': f"{mobility:.2f}%",
        'charge_plot': f"data:image/png;base64,{img_buffers['charge_plot']}",
        'temp_plot': f"data:image/png;base64,{img_buffers['temp_plot']}",
        'aa_plot': f"data:image/png;base64,{img_buffers['aa_plot']}",
        'amino_acid_composition': aa_rows,
        'amino_acid_charges': aa_charge,
        'charge_contributors': analyzer.get_charge_contributors(electrophoresis_ph),
        'physical_properties': analyzer.get_physical_properties(),
        'current_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pdb_analysis': pdb_analysis
    }
    
    # 添加PTM信息
    if analyzer.ptm_handler:
        context['ptm_config'] = analyzer.ptm_handler.get_ptm_summary()
    
    # 加载模板
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('report_template.html')
    
    # 渲染并保存
    html_content = template.render(context)
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file