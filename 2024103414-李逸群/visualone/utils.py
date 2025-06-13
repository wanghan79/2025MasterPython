import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PDBIO, Select

class LigandSelect(Select):
    """Select ligand atoms from PDB file"""
    def __init__(self, ligand_residues):
        self.ligand_residues = ligand_residues
        
    def accept_residue(self, residue):
        return residue in self.ligand_residues

class BindingSiteSelect(Select):
    """Select binding site atoms from PDB file"""
    def __init__(self, binding_site_atoms):
        self.binding_site_atoms = binding_site_atoms
        
    def accept_atom(self, atom):
        return atom in self.binding_site_atoms

def calculate_aromatic_rings(mol):
    """Calculate aromatic rings in a molecule"""
    rings = []
    for ring in mol.GetSSSR():
        if ring.GetNumAtoms() >= 5 and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            # Calculate ring center
            ring_atoms = [mol.GetConformer().GetAtomPosition(idx) for idx in ring]
            ring_center = np.mean(ring_atoms, axis=0)
            ring_normal = calculate_ring_normal(ring_atoms)
            rings.append((ring_center, ring_normal))
    return rings

def calculate_ring_normal(ring_atoms):
    """Calculate the normal vector of a ring"""
    # Convert to numpy array
    points = np.array(ring_atoms)
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Calculate the covariance matrix
    cov = np.cov(centered_points.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return normal

def calculate_pi_stacking_score(ring1_center, ring1_normal, ring2_center, ring2_normal):
    """Calculate π-π stacking score between two aromatic rings"""
    # Calculate distance between ring centers
    distance = np.linalg.norm(ring1_center - ring2_center)
    
    # Calculate angle between ring normals
    dot_product = np.dot(ring1_normal, ring2_normal)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Convert to degrees
    angle_deg = np.degrees(angle)
    
    # π-π stacking typically occurs at 3.5-4.5 Å with parallel or perpendicular rings
    # Parallel stacking: angle close to 0° or 180°
    # T-shaped stacking: angle close to 90°
    
    # Distance score (1.0 at optimal distance, decreasing as distance deviates)
    distance_score = 1.0 - abs(distance - 4.0) / 1.5
    distance_score = max(0.0, min(1.0, distance_score))
    
    # Angle score (1.0 at optimal angles, decreasing as angle deviates)
    parallel_score = 1.0 - min(angle_deg, 180 - angle_deg) / 30.0
    t_shaped_score = 1.0 - abs(angle_deg - 90) / 30.0
    angle_score = max(parallel_score, t_shaped_score)
    angle_score = max(0.0, min(1.0, angle_score))
    
    # Combined score
    score = distance_score * angle_score
    
    return score

def save_feature_image(feature_grid, feature_name, output_path, cmap='viridis'):
    """Save a feature grid as an image"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.imshow(feature_grid, cmap=cmap, interpolation='nearest')
    plt.colorbar(label=feature_name)
    plt.title(f"Protein-Ligand {feature_name}")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()