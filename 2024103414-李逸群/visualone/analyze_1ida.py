import os
import sys
from protein_ligand_analysis import ProteinLigandAnalysis

def main():
    # 设置路径 - 修改为直接使用当前目录下的1ida.pdb
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 不再使用父目录
    # parent_dir = os.path.dirname(current_dir)
    # pdb_file = os.path.join(parent_dir, "1ida.pdb")
    
    # 直接使用当前目录下的1ida.pdb
    pdb_file = os.path.join(current_dir, "1ida.pdb")
    output_dir = os.path.join(current_dir, "1ida_analysis")
    
    # 检查PDB文件是否存在
    if not os.path.exists(pdb_file):
        print(f"错误：未找到PDB文件 {pdb_file}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行分析
    analyzer = ProteinLigandAnalysis(
        pdb_file=pdb_file,
        output_dir=output_dir,
        patch_radius=10.0
    )
    
    analyzer.run_analysis()
    
    print("\nHIV-2蛋白酶(1IDA)与抑制剂的分析完成！")
    print("此分析包括：")
    print("1. 蛋白质与配体之间的形状互补性")
    print("2. 静电相互作用")
    print("3. 疏水相互作用")
    print("4. 氢键")
    print("5. π-π相互作用")
    print(f"\n结果保存在 {output_dir}")

if __name__ == "__main__":
    main()