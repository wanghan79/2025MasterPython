import os
import sys
import argparse
from protein_ligand_analysis import ProteinLigandAnalysis

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Protein-Ligand Interaction Analysis')
    parser.add_argument('--pdb', type=str, required=True, help='Path to PDB file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--radius', type=float, default=10.0, help='Binding site radius (Å)')
    args = parser.parse_args()
    
    # Check if PDB file exists
    if not os.path.exists(args.pdb):
        print(f"Error: PDB file {args.pdb} not found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run analysis
    analyzer = ProteinLigandAnalysis(
        pdb_file=args.pdb,
        output_dir=args.output,
        patch_radius=args.radius
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()