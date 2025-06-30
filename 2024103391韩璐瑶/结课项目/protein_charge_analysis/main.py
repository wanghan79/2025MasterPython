import argparse
import os
import sys
import csv
import time
from datetime import datetime
from charge_analyzer import ProteinChargeAnalyzer
from pdb_processor import PDBProcessor
from ptm_handler import PTMHandler
from report_generator import generate_html_report

def load_ptm_config(config_path):
    """加载PTM配置文件"""
    ptm_config = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row['position'] = int(row['position'])
                    row['effect'] = float(row['effect'])
                    ptm_config.append(row)
                except ValueError:
                    print(f"警告: 跳过无效PTM配置行: {row}")
    return ptm_config

def main():
    parser = argparse.ArgumentParser(description='高级蛋白质电荷分析工具', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--sequence', type=str, help='蛋白质序列')
    parser.add_argument('-f', '--fasta', type=str, help='FASTA文件路径')
    parser.add_argument('-p', '--pdb', type=str, help='PDB文件路径')
    parser.add_argument('-c', '--chain', type=str, default='A', help='PDB链ID')
    parser.add_argument('-o', '--output', type=str, default=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html', 
                        help='输出报告文件名')
    parser.add_argument('--ph', type=float, default=8.6, help='电泳pH条件')
    parser.add_argument('--temperature', type=float, default=25.0, help='温度(℃)')
    parser.add_argument('--ionic_strength', type=float, default=0.15, help='离子强度(M)')
    parser.add_argument('--plot', action='store_true', help='显示电荷分布图')
    parser.add_argument('--custom_pka', type=str, help='自定义pKa值CSV文件路径')
    parser.add_argument('--ptm_config', type=str, default='data/ptm_config.csv', 
                        help='翻译后修饰配置文件路径')
    parser.add_argument('--pymol', action='store_true', help='生成PyMOL可视化脚本')
    
    args = parser.parse_args()

    # 获取蛋白质序列
    sequence = args.sequence
    if args.fasta:
        try:
            with open(args.fasta, 'r') as f:
                sequence = ''.join([line.strip() for line in f.readlines() if not line.startswith('>')])
        except Exception as e:
            print(f"读取FASTA文件错误: {e}")
            sys.exit(1)
    
    if args.pdb:
        try:
            pdb_processor = PDBProcessor(args.pdb)
            sequence = pdb_processor.get_sequence(args.chain)
            print(f"从PDB文件 {args.pdb} 链 {args.chain} 提取序列: {sequence[:20]}... (共{len(sequence)}个氨基酸)")
        except Exception as e:
            print(f"处理PDB文件错误: {e}")
            sys.exit(1)
    
    if not sequence:
        print("错误: 必须提供序列、FASTA文件或PDB文件")
        parser.print_help()
        sys.exit(1)

    # 加载自定义pKa值
    custom_pka = None
    if args.custom_pka:
        try:
            custom_pka = {}
            with open(args.custom_pka, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    aa = row['amino_acid']
                    custom_pka[aa] = {
                        'pKa': float(row['pKa']),
                        'type': row['type']
                    }
            print(f"已加载自定义pKa值: {len(custom_pka)}个氨基酸")
        except Exception as e:
            print(f"加载自定义pKa值错误: {e}")
            sys.exit(1)
    
    # 加载PTM配置
    ptm_config = load_ptm_config(args.ptm_config)
    ptm_handler = PTMHandler(ptm_config)
    
    # 初始化分析器
    analyzer = ProteinChargeAnalyzer(
        sequence, 
        custom_pka=custom_pka,
        ptm_handler=ptm_handler,
        temperature=args.temperature,
        ionic_strength=args.ionic_strength
    )
    
    print("\n" + "="*50)
    print(f"分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"蛋白质序列: {sequence[:20]}... (共{len(sequence)}个氨基酸)")
    
    # 执行分析
    start_time = time.time()
    pI = analyzer.find_pI()
    analyzer.generate_charge_profile()
    analyzer.analyze_temperature_effect()
    
    # 预测电泳行为
    charge_at_ph = analyzer.calculate_net_charge(args.ph)
    direction, mobility = analyzer.predict_electrophoresis(args.ph)
    
    # 计算氨基酸特性
    aa_composition = analyzer.get_amino_acid_composition()
    aa_charge = analyzer.get_amino_acid_charge_at_ph(args.ph)
    charge_contributors = analyzer.get_charge_contributors(args.ph)
    
    # 生成PDB分析结果
    pdb_analysis = None
    if args.pdb:
        pdb_analysis = pdb_processor.analyze_structure()
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"分析完成! 耗时: {elapsed_time:.2f}秒")
    print("="*50 + "\n")
    
    # 打印关键结果
    print(f"计算等电点(pI): {pI}")
    print(f"在pH={args.ph}条件下净电荷: {charge_at_ph:.4f}")
    print(f"预测电泳迁移方向: {direction} (迁移率: {mobility:.2f}%)")
    
    # 生成报告
    generate_html_report(
        analyzer, 
        args.ph, 
        direction, 
        mobility,
        pdb_analysis=pdb_analysis,
        output_file=args.output
    )
    print(f"分析报告已保存至: {os.path.abspath(args.output)}")
    
    # 生成PyMOL脚本
    if args.pdb and args.pymol:
        pymol_script = pdb_processor.generate_pymol_script(analyzer, args.ph)
        script_name = os.path.splitext(args.output)[0] + "_charge.py"
        with open(script_name, 'w') as f:
            f.write(pymol_script)
        print(f"PyMOL可视化脚本已生成: {script_name}")
        print("提示: 在PyMOL中运行 'run {script_name}' 查看电荷表面")
    
    # 显示图表
    if args.plot:
        analyzer.plot_charge_distribution()
        analyzer.plot_temperature_effect()
        analyzer.plot_aa_composition()

if __name__ == '__main__':
    main()