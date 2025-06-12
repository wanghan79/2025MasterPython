import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from Bio.PDB import PDBParser, Selection, Select
from Bio.PDB.DSSP import DSSP
import mdtraj as md
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.spatial.distance import cdist
import tifffile
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
# 修改导入方式
# 从相对路径导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.utils import merge_chains
except ImportError:
    # 如果导入失败，定义一个简单的替代函数
    def merge_chains(structure):
        """简单的合并链函数，如果原始函数不可用"""
        return structure

class ProteinLigandAnalysis:
    def __init__(self, pdb_file, output_dir="output", patch_radius=10.0):
        """
        Initialize the protein-ligand analysis tool
        
        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file
        output_dir : str
            Directory to save output files
        patch_radius : float
            Radius around the binding site to consider for analysis
        """
        self.pdb_file = pdb_file
        self.output_dir = output_dir
        self.patch_radius = patch_radius
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
        
        # Parse PDB file
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("structure", pdb_file)
        
        # Extract protein and ligand
        self.extract_protein_ligand()
        
        # Features dictionary
        self.features = {}
        
    def extract_protein_ligand(self):
        """Extract protein and ligand from the PDB file"""
        # Identify ligand (HETATM records that are not water or standard residues)
        ligand_residues = []
        protein_chains = []
        
        # Standard amino acids and nucleotides
        standard_residues = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C', 'U', 'HOH', 'WAT'
        }
        
        for model in self.structure:
            for chain in model:
                is_protein_chain = True
                for residue in chain:
                    res_id = residue.get_id()
                    # Check if it's a hetero residue and not water
                    if res_id[0].startswith('H_') and residue.get_resname() not in standard_residues:
                        ligand_residues.append(residue)
                        is_protein_chain = False
                
                if is_protein_chain and len(list(chain.get_residues())) > 0:
                    protein_chains.append(chain.get_id())
        
        if not ligand_residues:
            # If no HETATM found, try to find ligand by looking at non-standard residues
            for model in self.structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_resname() not in standard_residues and residue.get_resname() != 'HOH':
                            ligand_residues.append(residue)
        
        if not ligand_residues:
            raise ValueError("No ligand found in the PDB file")
        
        self.ligand_residues = ligand_residues
        self.protein_chains = protein_chains
        
        # Get ligand atoms
        self.ligand_atoms = []
        for residue in ligand_residues:
            for atom in residue:
                self.ligand_atoms.append(atom)
        
        # Get protein atoms
        self.protein_atoms = []
        for model in self.structure:
            for chain in model:
                if chain.get_id() in protein_chains:
                    for residue in chain:
                        if residue.get_resname() in standard_residues:
                            for atom in residue:
                                self.protein_atoms.append(atom)
        
        print(f"Found {len(self.ligand_atoms)} ligand atoms and {len(self.protein_atoms)} protein atoms")
        
        # Get binding site atoms (protein atoms within patch_radius of any ligand atom)
        self.binding_site_atoms = []
        ligand_coords = np.array([atom.get_coord() for atom in self.ligand_atoms])
        
        for atom in self.protein_atoms:
            atom_coord = atom.get_coord()
            distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
            if np.min(distances) <= self.patch_radius:
                self.binding_site_atoms.append(atom)
        
        print(f"Found {len(self.binding_site_atoms)} binding site atoms")
        
    def prepare_rdkit_molecules(self):
        """Prepare RDKit molecules for protein and ligand"""
        # 创建一个临时PDB文件，只包含配体
        ligand_pdb = os.path.join(self.output_dir, "ligand.pdb")
        with open(ligand_pdb, 'w') as f:
            # 使用PDBIO来写入PDB文件
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(self.structure)
            
            # 定义一个选择器类来只选择配体残基
            class LigandSelect(Select):
                def __init__(self, ligand_residues):
                    self.ligand_residues = ligand_residues
                
                def accept_residue(self, residue):
                    return residue in self.ligand_residues
            
            # 使用选择器写入配体
            io.save(f, LigandSelect(self.ligand_residues))
        
        # 加载配体到RDKit
        self.ligand_mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False)
        if self.ligand_mol is None:
            print("警告：无法从配体PDB创建RDKit分子")
            # 尝试从SMILES创建分子（如果PDB中有）
            # 这是一个后备方案，可能不适用于所有配体
            self.ligand_mol = Chem.MolFromSmiles('C1=CC=CC=C1')  # 默认为苯
        else:
            # 如果不存在，添加氢原子
            self.ligand_mol = Chem.AddHs(self.ligand_mol)
            # 如果不存在，生成3D坐标
            AllChem.EmbedMolecule(self.ligand_mol)
        
        # 创建一个临时PDB文件，只包含结合位点
        binding_site_pdb = os.path.join(self.output_dir, "binding_site.pdb")
        with open(binding_site_pdb, 'w') as f:
            # 使用PDBIO来写入PDB文件
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(self.structure)
            
            # 定义一个选择器类来只选择结合位点原子
            class BindingSiteSelect(Select):
                def __init__(self, binding_site_atoms):
                    self.binding_site_atoms = binding_site_atoms
                
                def accept_atom(self, atom):
                    return atom in self.binding_site_atoms
            
            # 使用选择器写入结合位点
            io.save(f, BindingSiteSelect(self.binding_site_atoms))
        
        # 加载结合位点到RDKit
        self.binding_site_mol = Chem.MolFromPDBFile(binding_site_pdb, removeHs=False)
        if self.binding_site_mol is None:
            print("警告：无法从结合位点PDB创建RDKit分子")
        else:
            self.binding_site_mol = Chem.AddHs(self.binding_site_mol)
    
    def compute_shape_complementarity(self):
        """Compute shape complementarity between protein and ligand"""
        # Calculate shape index for binding site atoms
        binding_site_coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        ligand_coords = np.array([atom.get_coord() for atom in self.ligand_atoms])
        
        # Calculate distance matrix
        dist_matrix = cdist(binding_site_coords, ligand_coords)
        
        # Calculate shape complementarity score
        # Lower distance means better shape complementarity
        min_distances = np.min(dist_matrix, axis=1)
        shape_scores = np.exp(-min_distances / 2.0)  # Exponential decay with distance
        
        # Normalize scores
        shape_scores = (shape_scores - np.min(shape_scores)) / (np.max(shape_scores) - np.min(shape_scores))
        
        self.features['shape_complementarity'] = shape_scores
        return shape_scores
    
    def compute_electrostatics(self):
        """Compute electrostatic interactions between protein and ligand"""
        # Define partial charges for common atoms
        partial_charges = {
            'O': -0.5, 'N': -0.5, 'S': -0.3,
            'C': 0.1, 'H': 0.1, 'P': 0.5,
            'F': -0.3, 'CL': -0.3, 'BR': -0.2, 'I': -0.1
        }
        
        binding_site_coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        binding_site_elements = np.array([atom.element for atom in self.binding_site_atoms])
        
        ligand_coords = np.array([atom.get_coord() for atom in self.ligand_atoms])
        ligand_elements = np.array([atom.element for atom in self.ligand_atoms])
        
        # Calculate charges
        binding_site_charges = np.array([partial_charges.get(elem.upper(), 0.0) for elem in binding_site_elements])
        ligand_charges = np.array([partial_charges.get(elem.upper(), 0.0) for elem in ligand_elements])
        
        # Calculate distance matrix
        dist_matrix = cdist(binding_site_coords, ligand_coords)
        
        # Calculate electrostatic interaction (Coulomb's law)
        # 修改这里：使用 self.binding_site_atoms 而不是 binding_site_atoms
        electrostatic_scores = np.zeros(len(self.binding_site_atoms))
        
        for i in range(len(self.binding_site_atoms)):
            for j in range(len(self.ligand_atoms)):
                if dist_matrix[i, j] > 0:
                    # Coulomb's law: F = k * q1 * q2 / r^2
                    # We use a simplified version without the constant k
                    electrostatic_scores[i] += binding_site_charges[i] * ligand_charges[j] / (dist_matrix[i, j] ** 2)
        
        # Normalize scores
        if np.max(electrostatic_scores) != np.min(electrostatic_scores):
            electrostatic_scores = (electrostatic_scores - np.min(electrostatic_scores)) / (np.max(electrostatic_scores) - np.min(electrostatic_scores))
        else:
            electrostatic_scores = np.zeros_like(electrostatic_scores)
        
        self.features['electrostatics'] = electrostatic_scores
        return electrostatic_scores
    
    def compute_hydrophobicity(self):
        """Compute hydrophobic interactions between protein and ligand"""
        # Kyte & Doolittle hydrophobicity scale
        hydrophobicity_scale = {
            'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
            'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
            'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
        }
        
        # Calculate hydrophobicity for binding site atoms
        hydrophobic_scores = np.zeros(len(self.binding_site_atoms))
        
        for i, atom in enumerate(self.binding_site_atoms):
            residue = atom.get_parent()
            res_name = residue.get_resname()
            hydrophobicity = hydrophobicity_scale.get(res_name, 0.0)
            
            # Assign hydrophobicity to atom based on atom type
            # Carbon atoms get the full hydrophobicity value
            if atom.element == 'C':
                hydrophobic_scores[i] = hydrophobicity
            # Polar atoms (O, N, S) get negative values
            elif atom.element in ['O', 'N', 'S']:
                hydrophobic_scores[i] = -abs(hydrophobicity)
            # Other atoms get zero
            else:
                hydrophobic_scores[i] = 0.0
        
        # Normalize scores
        if np.max(hydrophobic_scores) != np.min(hydrophobic_scores):
            hydrophobic_scores = (hydrophobic_scores - np.min(hydrophobic_scores)) / (np.max(hydrophobic_scores) - np.min(hydrophobic_scores))
        else:
            hydrophobic_scores = np.zeros_like(hydrophobic_scores)
        
        self.features['hydrophobicity'] = hydrophobic_scores
        return hydrophobic_scores
    
    def compute_hydrogen_bonds(self):
        """Compute hydrogen bonds between protein and ligand"""
        # Define hydrogen bond donors and acceptors
        donors = ['N', 'O']  # Atoms that can donate H-bonds
        acceptors = ['O', 'N', 'S']  # Atoms that can accept H-bonds
        
        binding_site_coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        binding_site_elements = np.array([atom.element for atom in self.binding_site_atoms])
        
        ligand_coords = np.array([atom.get_coord() for atom in self.ligand_atoms])
        ligand_elements = np.array([atom.element for atom in self.ligand_atoms])
        
        # Calculate distance matrix
        dist_matrix = cdist(binding_site_coords, ligand_coords)
        
        # Calculate hydrogen bond scores
        # 修改这里：使用 self.binding_site_atoms 而不是 binding_site_atoms
        hbond_scores = np.zeros(len(self.binding_site_atoms))
        
        for i in range(len(self.binding_site_atoms)):
            if binding_site_elements[i] in donors + acceptors:
                for j in range(len(self.ligand_atoms)):
                    if ligand_elements[j] in acceptors + donors:
                        # Check if distance is within H-bond range (2.5-3.5 Å)
                        if 2.5 <= dist_matrix[i, j] <= 3.5:
                            # Simple scoring: closer to 2.8Å is better
                            hbond_scores[i] += 1.0 - abs(dist_matrix[i, j] - 2.8) / 0.7
        
        # Normalize scores
        if np.max(hbond_scores) != np.min(hbond_scores):
            hbond_scores = (hbond_scores - np.min(hbond_scores)) / (np.max(hbond_scores) - np.min(hbond_scores))
        else:
            hbond_scores = np.zeros_like(hbond_scores)
        
        self.features['hydrogen_bonds'] = hbond_scores
        return hbond_scores
    
    def compute_pi_pi_interactions(self):
        """Compute π-π interactions between protein and ligand"""
        # Prepare RDKit molecules if not already done
        if not hasattr(self, 'ligand_mol') or not hasattr(self, 'binding_site_mol'):
            self.prepare_rdkit_molecules()
        
        # Define aromatic residues
        aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        
        # Identify aromatic rings in the ligand
        ligand_aromatic_rings = []
        if self.ligand_mol:
            # 修改这里：使用 Chem.GetSSSR 而不是 self.ligand_mol.GetSSSR
            from rdkit.Chem import GetSSSR
            for ring in GetSSSR(self.ligand_mol):
                # 修改这里：使用 len(ring) 而不是 ring.GetNumAtoms()
                if len(ring) >= 5 and all(self.ligand_mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                    # Calculate ring center
                    ring_atoms = [self.ligand_mol.GetConformer().GetAtomPosition(idx) for idx in ring]
                    ring_center = np.mean(ring_atoms, axis=0)
                    ligand_aromatic_rings.append(ring_center)
        
        # Identify aromatic rings in the binding site
        binding_site_aromatic_centers = []
        binding_site_aromatic_indices = []
        
        for i, atom in enumerate(self.binding_site_atoms):
            residue = atom.get_parent()
            res_name = residue.get_resname()
            
            if res_name in aromatic_residues:
                # For simplicity, use the center of the residue as the ring center
                # In a more sophisticated implementation, you would identify the actual ring atoms
                if atom.name == 'CG' and res_name in ['PHE', 'TYR']:
                    binding_site_aromatic_centers.append(atom.get_coord())
                    binding_site_aromatic_indices.append(i)
                elif atom.name == 'CG' and res_name == 'TRP':
                    binding_site_aromatic_centers.append(atom.get_coord())
                    binding_site_aromatic_indices.append(i)
                elif atom.name == 'CG' and res_name == 'HIS':
                    binding_site_aromatic_centers.append(atom.get_coord())
                    binding_site_aromatic_indices.append(i)
        
        # Calculate π-π interaction scores
        pi_pi_scores = np.zeros(len(self.binding_site_atoms))
        
        for i, center in enumerate(binding_site_aromatic_centers):
            atom_idx = binding_site_aromatic_indices[i]
            for ligand_center in ligand_aromatic_rings:
                # Calculate distance between ring centers
                distance = np.linalg.norm(center - ligand_center)
                
                # π-π stacking typically occurs at 3.5-4.5 Å
                if 3.5 <= distance <= 5.5:
                    # Simple scoring: closer to 4.0Å is better
                    pi_pi_scores[atom_idx] += 1.0 - abs(distance - 4.0) / 1.5
        
        # Normalize scores
        if np.max(pi_pi_scores) != np.min(pi_pi_scores):
            pi_pi_scores = (pi_pi_scores - np.min(pi_pi_scores)) / (np.max(pi_pi_scores) - np.min(pi_pi_scores))
        else:
            pi_pi_scores = np.zeros_like(pi_pi_scores)
        
        self.features['pi_pi_interactions'] = pi_pi_scores
        return pi_pi_scores
    
    def compute_halogen_bonds(self):
        halogens = ['CL', 'BR', 'I', 'F']
        acceptors = ['O', 'N']
        ligand_coords = np.array([atom.get_coord() for atom in self.ligand_atoms])
        ligand_elements = np.array([atom.element.upper() for atom in self.ligand_atoms])
        binding_coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        binding_elements = np.array([atom.element.upper() for atom in self.binding_site_atoms])
        dist_matrix = cdist(ligand_coords, binding_coords)
        halogen_scores = np.zeros(len(self.binding_site_atoms))
        for i, (lig_coord, lig_elem) in enumerate(zip(ligand_coords, ligand_elements)):
            if lig_elem in halogens:
                for j, (bind_coord, bind_elem) in enumerate(zip(binding_coords, binding_elements)):
                    if bind_elem in acceptors:
                        dist = np.linalg.norm(lig_coord - bind_coord)
                        if dist < 3.5:
                            halogen_scores[j] += 1.0 - abs(dist - 3.0) / 0.5
        halogen_scores = (halogen_scores - np.min(halogen_scores)) / (np.max(halogen_scores) - np.min(halogen_scores) + 1e-6)
        self.features['halogen_bonds'] = halogen_scores
        return halogen_scores

    def compute_cation_pi(self):
        cationic_residues = ['ARG', 'LYS', 'HIS']
        aromatic_residues = ['PHE', 'TYR', 'TRP']
        coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        elements = np.array([atom.element.upper() for atom in self.binding_site_atoms])
        residues = [atom.get_parent().get_resname() for atom in self.binding_site_atoms]
        scores = np.zeros(len(self.binding_site_atoms))
        for i, (elem, res, coord) in enumerate(zip(elements, residues, coords)):
            if res in cationic_residues:
                for j, (res2, coord2) in enumerate(zip(residues, coords)):
                    if res2 in aromatic_residues:
                        dist = np.linalg.norm(coord - coord2)
                        if 3.0 <= dist <= 6.0:
                            scores[i] += 1.0 - abs(dist - 4.5) / 1.5
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
        self.features['cation_pi'] = scores
        return scores

    def compute_salt_bridges(self):
        acidic = ['ASP', 'GLU']
        basic = ['ARG', 'LYS', 'HIS']
        coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        residues = [atom.get_parent().get_resname() for atom in self.binding_site_atoms]
        scores = np.zeros(len(self.binding_site_atoms))
        for i, (res1, coord1) in enumerate(zip(residues, coords)):
            if res1 in acidic:
                for j, (res2, coord2) in enumerate(zip(residues, coords)):
                    if res2 in basic:
                        dist = np.linalg.norm(coord1 - coord2)
                        if dist < 4.0:
                            scores[i] += 1.0 - abs(dist - 3.0) / 1.0
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
        self.features['salt_bridges'] = scores
        return scores

    def compute_hydrophobic_pockets(self):
        hydrophobic_residues = ['VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO', 'ALA']
        scores = np.zeros(len(self.binding_site_atoms))
        residue_names = [atom.get_parent().get_resname() for atom in self.binding_site_atoms]
        for i, res in enumerate(residue_names):
            if res in hydrophobic_residues:
                scores[i] = 1.0
        self.features['hydrophobic_pocket'] = scores
        return scores

    def compute_all_features(self):
        """Compute all features"""
        print("Computing shape complementarity...")
        self.compute_shape_complementarity()
        
        print("Computing electrostatics...")
        self.compute_electrostatics()
        
        print("Computing hydrophobicity...")
        self.compute_hydrophobicity()
        
        print("Computing hydrogen bonds...")
        self.compute_hydrogen_bonds()
        
        print("Computing π-π interactions...")
        self.compute_pi_pi_interactions()
        print("Computing halogen bonds...")
        self.compute_halogen_bonds()
        print("Computing cation-π interactions...")
        self.compute_cation_pi()
        print("Computing salt bridges...")
        self.compute_salt_bridges()
        print("Computing hydrophobic pockets...")
        self.compute_hydrophobic_pockets()
        
        # Stack all features into a single array
        feature_names = list(self.features.keys())
        feature_arrays = [self.features[name] for name in feature_names]
        
        # Create a stacked array
        self.stacked_features = np.column_stack(feature_arrays)
        
        # Save features as numpy array
        np.save(os.path.join(self.output_dir, "features", "stacked_features.npy"), self.stacked_features)
        
        # Save feature names
        with open(os.path.join(self.output_dir, "features", "feature_names.txt"), 'w') as f:
            for name in feature_names:
                f.write(name + '\n')
        
        return self.features
    
    def visualize_features(self, grid_size=32):
        """Visualize features as heatmaps"""
        if not hasattr(self, 'features') or len(self.features) == 0:
            self.compute_all_features()
        
        # Create a grid for visualization
        grid_x = np.linspace(0, grid_size-1, grid_size)
        grid_y = np.linspace(0, grid_size-1, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Get binding site coordinates
        binding_site_coords = np.array([atom.get_coord() for atom in self.binding_site_atoms])
        
        # Normalize coordinates to fit in the grid
        min_coords = np.min(binding_site_coords, axis=0)
        max_coords = np.max(binding_site_coords, axis=0)
        
        # Create a multi-channel TIFF image
        tiff_channels = []
        
        # Create a figure for each feature
        for feature_name, feature_values in self.features.items():
            # Create a grid for this feature
            feature_grid = np.zeros((grid_size, grid_size))
            
            # Map binding site atoms to grid cells
            for i, coord in enumerate(binding_site_coords):
                # Normalize coordinates to grid indices
                grid_i = int((coord[0] - min_coords[0]) / (max_coords[0] - min_coords[0]) * (grid_size-1))
                grid_j = int((coord[1] - min_coords[1]) / (max_coords[1] - min_coords[1]) * (grid_size-1))
                
                # Ensure indices are within bounds
                grid_i = max(0, min(grid_i, grid_size-1))
                grid_j = max(0, min(grid_j, grid_size-1))
                
                # Assign feature value to grid cell
                feature_grid[grid_j, grid_i] = feature_values[i]
            
            # Add to TIFF channels
            tiff_channels.append(feature_grid)
            
            # Create a heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(feature_grid, cmap='viridis', interpolation='nearest')
            plt.colorbar(label=feature_name)
            plt.title(f"Protein-Ligand {feature_name}")
            plt.xlabel("X Grid")
            plt.ylabel("Y Grid")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "visualization", f"{feature_name}.png"), dpi=300)
            plt.close()
        
        # Save multi-channel TIFF
        tiff_array = np.stack(tiff_channels, axis=0)
        tifffile.imwrite(os.path.join(self.output_dir, "features", "multi_channel_features.tiff"), tiff_array)
        
        # Create a combined visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (feature_name, feature_values) in enumerate(self.features.items()):
            if i < len(axes):
                # Create a grid for this feature
                feature_grid = np.zeros((grid_size, grid_size))
                
                # Map binding site atoms to grid cells
                for j, coord in enumerate(binding_site_coords):
                    # Normalize coordinates to grid indices
                    grid_i = int((coord[0] - min_coords[0]) / (max_coords[0] - min_coords[0]) * (grid_size-1))
                    grid_j = int((coord[1] - min_coords[1]) / (max_coords[1] - min_coords[1]) * (grid_size-1))
                    
                    # Ensure indices are within bounds
                    grid_i = max(0, min(grid_i, grid_size-1))
                    grid_j = max(0, min(grid_j, grid_size-1))
                    
                    # Assign feature value to grid cell
                    feature_grid[grid_j, grid_i] = feature_values[j]
                
                # Plot on the corresponding axis
                im = axes[i].imshow(feature_grid, cmap='viridis', interpolation='nearest')
                axes[i].set_title(f"Protein-Ligand {feature_name}")
                fig.colorbar(im, ax=axes[i])
        
        # Remove any unused axes
        for i in range(len(self.features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualization", "combined_features.png"), dpi=300)
        plt.close()
        
        # Create an interactive HTML visualization
        self.create_interactive_visualization()
    
    def create_interactive_visualization(self):
        """Create an interactive HTML visualization"""
        # Create a simple HTML file with interactive visualization
        html_file = os.path.join(self.output_dir, "visualization", "interactive.html")
        
        # Get feature names and create JavaScript arrays
        feature_names = list(self.features.keys())
        js_arrays = []
        
        for feature_name in feature_names:
            # Convert feature values to a JavaScript array
            js_array = "["
            for value in self.features[feature_name]:
                js_array += f"{value:.4f},"
            js_array = js_array[:-1] + "]"  # Remove last comma and close array
            js_arrays.append(js_array)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Protein-Ligand Interaction Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; flex-wrap: wrap; }}
                .plot {{ width: 600px; height: 500px; margin: 10px; }}
                .header {{ width: 100%; text-align: center; margin-bottom: 20px; }}
                .description {{ width: 100%; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Protein-Ligand Interaction Analysis</h1>
                <p>PDB File: {os.path.basename(self.pdb_file)}</p>
            </div>
            
            <div class="description">
                <h2>Feature Descriptions</h2>
                <ul>
                    <li><strong>Shape Complementarity</strong>: Measures how well the ligand fits into the binding site.</li>
                    <li><strong>Electrostatics</strong>: Measures electrostatic interactions between charged atoms.</li>
                    <li><strong>Hydrophobicity</strong>: Measures hydrophobic interactions in the binding site.</li>
                    <li><strong>Hydrogen Bonds</strong>: Identifies potential hydrogen bonds between protein and ligand.</li>
                    <li><strong>π-π Interactions</strong>: Identifies potential π-π stacking interactions between aromatic rings.</li>
                </ul>
            </div>
            
            <div class="container">
                <div id="combined-plot" class="plot"></div>
        """
        
        # Add individual plots for each feature
        for i, feature_name in enumerate(feature_names):
            html_content += f'<div id="plot-{i}" class="plot"></div>\n'
        
        # Add JavaScript for plotting
        html_content += """
            </div>
            
            <script>
                // Feature data
        """
        
        # Add feature data
        for i, (feature_name, js_array) in enumerate(zip(feature_names, js_arrays)):
            html_content += f"var {feature_name.replace('-', '_')} = {js_array};\n"
        
        # Add plotting code
        html_content += """
                // Create a combined plot
                var combinedData = [];
        """
        
        # Add traces for combined plot
        for i, feature_name in enumerate(feature_names):
            safe_name = feature_name.replace('-', '_')
            html_content += f"""
                combinedData.push({{
                    y: {safe_name},
                    type: 'box',
                    name: '{feature_name}'
                }});
            """
        
        # Complete the combined plot
        html_content += """
                var combinedLayout = {
                    title: 'Feature Distribution Comparison',
                    yaxis: {
                        title: 'Feature Value'
                    }
                };
                
                Plotly.newPlot('combined-plot', combinedData, combinedLayout);
        """
        
        # Add individual plots
        for i, feature_name in enumerate(feature_names):
            safe_name = feature_name.replace('-', '_')
            html_content += f"""
                // Create histogram for {feature_name}
                var data_{i} = [{{
                    x: {safe_name},
                    type: 'histogram',
                    marker: {{
                        color: 'rgba(0, 100, 200, 0.7)'
                    }}
                }}];
                
                var layout_{i} = {{
                    title: '{feature_name} Distribution',
                    xaxis: {{
                        title: 'Feature Value'
                    }},
                    yaxis: {{
                        title: 'Count'
                    }}
                }};
                
                Plotly.newPlot('plot-{i}', data_{i}, layout_{i});
            """
        
        # Close HTML
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # Write HTML file
                # 完成HTML文件生成
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Interactive visualization saved to {html_file}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print(f"Analyzing protein-ligand interactions in {self.pdb_file}...")
        
        # Extract protein and ligand
        print("Extracting protein and ligand...")
        self.extract_protein_ligand()
        
        # Prepare RDKit molecules
        print("Preparing molecular representations...")
        self.prepare_rdkit_molecules()
        
        # Compute features
        print("Computing interaction features...")
        self.compute_all_features()
        
        # Visualize features
        print("Generating visualizations...")
        self.visualize_features()
        
        print("Analysis complete!")
        print(f"Results saved to {self.output_dir}")
        
        return self.features