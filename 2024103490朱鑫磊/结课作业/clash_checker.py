import logging
from typing import Optional

import torch
import torch.nn as nn

ID2TYPE = {0: "UNK", 1: "lig", 2: "prot", 3: "dna", 4: "rna"}

class Clash(nn.Module):
    def __init__(
        self,
        af3_clash_threshold=1.1, # af3的clash阈值
        vdw_clash_threshold=0.75,
        compute_af3_clash=True,
        compute_vdw_clash=True,
    ):
        super().__init__()
        self.af3_clash_threshold = af3_clash_threshold
        self.vdw_clash_threshold = vdw_clash_threshold
        self.compute_af3_clash = compute_af3_clash
        self.compute_vdw_clash = compute_vdw_clash

    def forward(
        self,
        pred_coordinate, # 原子坐标 [1, N_atom, 3]
        asym_id, # 标记每个氨基酸属于哪个链 [N_token, ]     int
        atom_to_token_idx, # 每个原子属于哪个残基 [N_atom] int
        mol_id: Optional[torch.Tensor] = None, # vwd
        elements_one_hot: Optional[torch.Tensor] = None, # vwd
    ):
        # 只输入蛋白质，所以其他分子类型全部设为 false
        is_protein = torch.ones_like(atom_to_token_idx, dtype=torch.bool)
        is_ligand = torch.zeros_like(is_protein, dtype=torch.bool)
        is_dna = torch.zeros_like(is_protein, dtype=torch.bool)
        is_rna = torch.zeros_like(is_protein, dtype=torch.bool)
      
        chain_info = self.get_chain_info(
            asym_id=asym_id,
            atom_to_token_idx=atom_to_token_idx,
            is_ligand=is_ligand,
            is_protein=is_protein,
            is_dna=is_dna,
            is_rna=is_rna,
        )

        return self._check_clash_per_chain_pairs(
            pred_coordinate=pred_coordinate, **chain_info
        )

    def get_chain_info(
        self,
        asym_id,
        atom_to_token_idx,
        is_ligand,
        is_protein,
        is_dna,
        is_rna,
        mol_id: Optional[torch.Tensor] = None, # vwd
        elements_one_hot: Optional[torch.Tensor] = None, # vwd
    ):
        # Get chain info
        asym_id = asym_id.long()
        # 每个链的残基掩码字典 直接通过 asym_id计算获得
        asym_id_to_asym_mask = {
            aid.item(): asym_id == aid for aid in torch.unique(asym_id)
        }
        N_chains = len(asym_id_to_asym_mask)
        # Make sure it is from 0 to N_chain-1
        assert N_chains == asym_id.max() + 1

        # Check and compute chain_types
        chain_types = []
        mol_id_to_asym_ids, asym_id_to_mol_id = {}, {}
        atom_type = (1 * is_ligand + 2 * is_protein + 3 * is_dna + 4 * is_rna).long()

        chain_types = ["prot"] * N_chains

        chain_info = {
            "N_chains": N_chains, # 链总数 int
            "atom_to_token_idx": atom_to_token_idx, # 每个原子是哪个残基 [N_atom] int
            "asym_id_to_asym_mask": asym_id_to_asym_mask, # 每个链的残基掩码字典，dict[链id] -> [N_token] (Bool)
            "atom_type": atom_type, # 每个原子的分子类型 [N_atom] int
            "mol_id": mol_id, # vdw 
            "elements_one_hot": elements_one_hot, # vdw
            "chain_types": chain_types,
        }

        return chain_info

    def get_chain_pair_violations(
        self,
        pred_coordinate,
        violation_type, # af3
        chain_1_mask, # [N_atom] bool
        chain_2_mask, # [N_atom] bool
        elements_one_hot: Optional[torch.Tensor] = None,
    ):

        chain_1_coords = pred_coordinate[chain_1_mask, :]  # 取出第一条链的原子
        chain_2_coords = pred_coordinate[chain_2_mask, :]  # 取出第二条链的原子
        pred_dist = torch.cdist(chain_1_coords, chain_2_coords) # 计算原子间的距离

        clash_per_atom_pair = (
            pred_dist < self.af3_clash_threshold # 若距离小于阈值，设为 true
        )  # [ N_atom_chain_1, N_atom_chain_2]
        clashed_col, clashed_row = torch.where(clash_per_atom_pair)
        clash_atom_pairs = torch.stack((clashed_col, clashed_row), dim=-1)

        return clash_atom_pairs

    def _check_clash_per_chain_pairs(
        self,
        pred_coordinate,
        atom_to_token_idx,
        N_chains,
        atom_type,
        chain_types,
        elements_one_hot,
        asym_id_to_asym_mask,
        mol_id: Optional[torch.Tensor] = None,
        asym_id_to_mol_id: Optional[torch.Tensor] = None,
    ):
        device = pred_coordinate.device
        N_sample = pred_coordinate.shape[0]

        #  初始化结果
        if self.compute_af3_clash:
            has_af3_clash_flag = torch.zeros(
                N_sample, N_chains, N_chains, device=device, dtype=torch.bool
            )
            af3_clash_details = torch.zeros(
                N_sample, N_chains, N_chains, 2, device=device
            )
        if self.compute_vdw_clash:
            has_vdw_clash_flag = torch.zeros(
                N_sample, N_chains, N_chains, device=device, dtype=torch.bool
            )
            vdw_clash_details = {}

        skipped_pairs = []
        for sample_id in range(N_sample):
            for i in range(N_chains):
                if chain_types[i] == "UNK":
                    continue
                atom_chain_mask_i = asym_id_to_asym_mask[i][atom_to_token_idx]
                N_chain_i = torch.sum(atom_chain_mask_i).item()
                for j in range(i + 1, N_chains):
                    if chain_types[j] == "UNK":
                        continue
                    chain_pair_type = set([chain_types[i], chain_types[j]])
                    # Skip potential bonded ligand to polymers
                    skip_bonded_ligand = False
                    if (
                        self.compute_vdw_clash
                        and "lig" in chain_pair_type
                        and len(chain_pair_type) > 1
                        and asym_id_to_mol_id[i] == asym_id_to_mol_id[j]
                    ):
                        common_mol_id = asym_id_to_mol_id[i]
                        logging.warning(
                            f"mol_id {common_mol_id} may contain bonded ligand to polymers"
                        )
                        skip_bonded_ligand = True
                        skipped_pairs.append((i, j))
                    atom_chain_mask_j = asym_id_to_asym_mask[j][atom_to_token_idx]
                    N_chain_j = torch.sum(atom_chain_mask_j).item()

                    if self.compute_vdw_clash and not skip_bonded_ligand:
                        vdw_clash_pairs = self.get_chain_pair_violations(
                            pred_coordinate=pred_coordinate[sample_id, :, :],
                            violation_type="vdw",
                            chain_1_mask=atom_chain_mask_i,
                            chain_2_mask=atom_chain_mask_j,
                            elements_one_hot=elements_one_hot,
                        )

                        if vdw_clash_pairs.shape[0] > 0:
                            vdw_clash_details[(sample_id, i, j)] = vdw_clash_pairs
                            has_vdw_clash_flag[sample_id, i, j] = True
                            has_vdw_clash_flag[sample_id, j, i] = True

                    if (
                        chain_types[i] == "lig" or chain_types[j] == "lig"
                    ):  # AF3 clash only consider polymer chains
                        continue

                    if self.compute_af3_clash:

                        # 根据遍历的 i 链和 j 链，计算链之间的原子对冲突情况
                        af3_clash_pairs = self.get_chain_pair_violations(
                            pred_coordinate=pred_coordinate[sample_id, :, :],
                            violation_type="af3",
                            chain_1_mask=atom_chain_mask_i, # 链上1的原子掩码
                            chain_2_mask=atom_chain_mask_j, # 链上2的原子掩码
                        )

                        total_clash = af3_clash_pairs.shape[0] # 统计total clash
                        relative_clash = total_clash / min(N_chain_i, N_chain_j) # 计算相对clash

                        # 计算链为i和j的时候的冲突情况 0 索引为 total clash , 1 索引为 relative_clash
                        af3_clash_details[sample_id, i, j, 0] = total_clash
                        af3_clash_details[sample_id, i, j, 1] = relative_clash

                        # 判断 i 和 j 是否有严重原子冲突
                        has_af3_clash_flag[sample_id, i, j] = (
                            total_clash > 100 or relative_clash > 0.5
                        )

                        # 镜像复制
                        af3_clash_details[sample_id, j, i, :] = af3_clash_details[
                            sample_id, i, j, :
                        ]

                        has_af3_clash_flag[sample_id, j, i] = has_af3_clash_flag[
                            sample_id, i, j
                        ]

        return {
            "summary": {
                "af3_clash": has_af3_clash_flag if self.compute_af3_clash else None,
            },
            "details": {
                "af3_clash": af3_clash_details if self.compute_af3_clash else None,
            },
        }
