from Bio.PDB import PDBParser, is_aa
import torch
from collections import defaultdict
from clash_checker import Clash
import time

def parse_pdb_for_clash(pdb_file: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)

    coords = []
    atom_to_token_idx = []
    is_polymer = []
    residue_list = []
    chain_id_map = {}
    asym_id_list = []
    chain_id_counter = 0

    chain_atom_counts = defaultdict(int) 

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id not in chain_id_map:
                chain_id_map[chain_id] = chain_id_counter
                chain_id_counter += 1
            asym_id_encoded = chain_id_map[chain_id]

            for res in chain:
                if not is_aa(res):  # 跳过非标准残基（非蛋白质）
                    continue

                token_idx = len(residue_list)
                residue_list.append(res)

                for atom in res:
                    coords.append(atom.coord)
                    atom_to_token_idx.append(token_idx)
                    is_polymer.append(True)
                    chain_atom_counts[chain_id] += 1 

                asym_id_list.append(asym_id_encoded)

    pred_coordinate = torch.tensor(coords).unsqueeze(0)  # [1, N_atom, 3]
    asym_id = torch.tensor(asym_id_list, dtype=torch.long)  # [N_token]
    atom_to_token_idx = torch.tensor(atom_to_token_idx, dtype=torch.long)  # [N_atom]
    is_polymer = torch.tensor(is_polymer, dtype=torch.bool)  # [N_atom]

    return pred_coordinate, asym_id, atom_to_token_idx, is_polymer, chain_atom_counts

def compute_clash_statistics_cpu(pdb_path):
    pred_coordinate, asym_id, atom_to_token_idx, is_polymer, chain_atom_counts = parse_pdb_for_clash(pdb_path)
    is_protein = torch.ones_like(is_polymer, dtype=torch.bool)

    clash_checker = Clash(compute_vdw_clash=False)

    start = time.time()
    results = clash_checker(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_to_token_idx,
    )
    end = time.time()
    print(f"cpu clash time:  {end - start}")

    af3_details = results["details"]["af3_clash"]
    N_chains = results["summary"]["af3_clash"].shape[1]

    total_clash = 0
    relative_clashes = []

    for i in range(N_chains):
        for j in range(i + 1, N_chains):
            clash_count = af3_details[0][i][j][0].item()
            rel_clash = af3_details[0][i][j][1].item()
            total_clash += clash_count
            relative_clashes.append(rel_clash)

    return total_clash, relative_clashes

def compute_clash_statistics_gpu(pdb_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_coordinate, asym_id, atom_to_token_idx, is_polymer, chain_atom_counts = parse_pdb_for_clash(pdb_path)
    pred_coordinate = pred_coordinate.to(device)
    asym_id = asym_id.to(device)
    atom_to_token_idx = atom_to_token_idx.to(device)
    is_polymer = is_polymer.to(device)

    clash_checker = Clash(compute_vdw_clash=False).to(device)

    start = time.time()
    results = clash_checker(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_to_token_idx,
    )
    end = time.time()
    print("gpu clash time:", end - start)
    # print("Using device:", pred_coordinate.device)


    af3_details = results["details"]["af3_clash"]
    N_chains = results["summary"]["af3_clash"].shape[1]

    total_clash = 0
    relative_clashes = []

    for i in range(N_chains):
        for j in range(i + 1, N_chains):
            clash_count = af3_details[0][i][j][0].item()
            rel_clash = af3_details[0][i][j][1].item()
            total_clash += clash_count
            relative_clashes.append(rel_clash)

    return total_clash, relative_clashes
