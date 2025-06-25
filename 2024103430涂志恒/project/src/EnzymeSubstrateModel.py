# coding=utf-8
# EnzymeSubstrateModel

import torch
import torch.nn as nn
from ProteinGNN import ProteinGNN
from MolecularAtomEncoder import MolecularAtomEncoder
from MolecularMotifEncoder import MolecularMotifEncoder
import json


class EnzymeSubstrateModel(nn.Module):
    def __init__(self,
                 protein_dim=256,
                 esm2_dim=1280,
                 atom_encoder_dim=256,
                 motif_encoder_dim=256,
                 hidden_dim=256,
                 output_dim=1):
        super().__init__()
        self.protein_gnn = ProteinGNN(out_dim=protein_dim)
        self.esm2_proj = nn.Linear(esm2_dim, protein_dim)
        self.atom_encoder = MolecularAtomEncoder(n_layers=4, d_model=atom_encoder_dim, n_head=8, d_ff=512)
        self.motif_encoder = MolecularMotifEncoder(
            vocab_size=len(json.load(open('data/motif_token.json', 'r'))), n_layers=4, d_model=motif_encoder_dim,
            n_head=8, d_ff=2048)

        # 门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(protein_dim * 2, protein_dim // 2),
            nn.LayerNorm(protein_dim // 2),
            nn.GELU(),
            nn.Linear(protein_dim // 2, protein_dim),
            nn.Sigmoid()
        )

        self.enzyme_attn = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=8,
            kdim=atom_encoder_dim,
            vdim=atom_encoder_dim,
            batch_first=True
        )

        self.substrate_attn = nn.MultiheadAttention(
            embed_dim=atom_encoder_dim,
            num_heads=8,
            kdim=protein_dim,
            vdim=protein_dim,
            batch_first=True
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(protein_dim + atom_encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, batch):
        protein_struct = self.protein_gnn(batch['enzyme']['structure'])
        protein_esm2 = self.esm2_proj(batch['enzyme']['esm2'])

        atom_repr, atom_gl_feat = self.atom_encoder(
            batch['substrate']['atom']['x'],
            batch['substrate']['atom']['adj'],
            batch['substrate']['atom']['dist'],
            batch['substrate']['atom']['mask']
        )
        # coding=utf-8
        # EnzymeSubstrateModel

        import torch
        import torch.nn as nn
        from ProteinGNN import ProteinGNN
        from MolecularAtomEncoder import MolecularAtomEncoder
        from MolecularMotifEncoder import MolecularMotifEncoder
        import json

        class EnzymeSubstrateModel(nn.Module):
            def __init__(self,
                         protein_dim=256,
                         esm2_dim=1280,
                         atom_encoder_dim=256,
                         motif_encoder_dim=256,
                         hidden_dim=256,
                         output_dim=1):
                super().__init__()
                self.protein_gnn = ProteinGNN(out_dim=protein_dim)
                self.esm2_proj = nn.Linear(esm2_dim, protein_dim)
                self.atom_encoder = MolecularAtomEncoder(n_layers=4, d_model=atom_encoder_dim, n_head=8, d_ff=512)
                self.motif_encoder = MolecularMotifEncoder(
                    vocab_size=len(json.load(open('data/motif_token.json', 'r'))), n_layers=4,
                    d_model=motif_encoder_dim,
                    n_head=8, d_ff=2048)

                # 门控机制
                self.fusion_gate = nn.Sequential(
                    nn.Linear(protein_dim * 2, protein_dim // 2),
                    nn.LayerNorm(protein_dim // 2),
                    nn.GELU(),
                    nn.Linear(protein_dim // 2, protein_dim),
                    nn.Sigmoid()
                )

                self.enzyme_attn = nn.MultiheadAttention(
                    embed_dim=protein_dim,
                    num_heads=8,
                    kdim=atom_encoder_dim,
                    vdim=atom_encoder_dim,
                    batch_first=True
                )

                self.substrate_attn = nn.MultiheadAttention(
                    embed_dim=atom_encoder_dim,
                    num_heads=8,
                    kdim=protein_dim,
                    vdim=protein_dim,
                    batch_first=True
                )

                # 特征融合
                self.fusion = nn.Sequential(
                    nn.Linear(protein_dim + atom_encoder_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                )

                # 输出层
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, output_dim)
                )

            def forward(self, batch):
                protein_struct = self.protein_gnn(batch['enzyme']['structure'])
                protein_esm2 = self.esm2_proj(batch['enzyme']['esm2'])

                atom_repr, atom_gl_feat = self.atom_encoder(
                    batch['substrate']['atom']['x'],
                    batch['substrate']['atom']['adj'],
                    batch['substrate']['atom']['dist'],
                    batch['substrate']['atom']['mask']
                )

                # batch_size, num_motifs, num_atoms = batch['substrate']['motif']['atom_map'].shape
                # atom_repr_padded = F.pad(atom_repr, (0, 0, 0, num_atoms - atom_repr.size(1)))
                # atomf = torch.bmm(
                #     batch['substrate']['motif']['atom_map'],
                #     atom_repr_padded
                # ) / batch['substrate']['motif']['sum_atoms']
                # motif_output, _, _, _ = self.motif_encoder(
                #     batch['substrate']['motif']['ids'],
                #     atomf,
                #     batch['substrate']['motif']['adj'],
                #     batch['substrate']['motif']['dist']
                # )

                # combined = torch.cat([
                #     protein_feat,
                #     motif_output[:, 0, :]
                # ], dim=1)

                # 门控机制酶特征融合
                fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat([protein_esm2, protein_struct], -1)))
                enzyme_feat = fusion_gate * protein_esm2 + (1 - fusion_gate) * protein_struct

                # 酶到底物的注意力
                enzyme_as_query = self.enzyme_attn(
                    query=enzyme_feat.unsqueeze(0),
                    key=atom_gl_feat.unsqueeze(0),
                    value=atom_gl_feat.unsqueeze(0)
                )[0].squeeze(0)

                # 底物到酶的注意力
                substrate_as_query = self.substrate_attn(
                    query=atom_gl_feat.unsqueeze(0),
                    key=enzyme_feat.unsqueeze(0),
                    value=enzyme_feat.unsqueeze(0)
                )[0].squeeze(0)

                # 对称交互
                combined = torch.cat([
                    enzyme_feat + 0.3 * enzyme_as_query,  # 酶特征+底物上下文
                    atom_gl_feat + 0.3 * substrate_as_query  # 底物特征+酶上下文
                ], dim=-1)

                return self.classifier(self.fusion(combined))

        # batch_size, num_motifs, num_atoms = batch['substrate']['motif']['atom_map'].shape
        # atom_repr_padded = F.pad(atom_repr, (0, 0, 0, num_atoms - atom_repr.size(1)))
        # atomf = torch.bmm(
        #     batch['substrate']['motif']['atom_map'],
        #     atom_repr_padded
        # ) / batch['substrate']['motif']['sum_atoms']
        # motif_output, _, _, _ = self.motif_encoder(
        #     batch['substrate']['motif']['ids'],
        #     atomf,
        #     batch['substrate']['motif']['adj'],
        #     batch['substrate']['motif']['dist']
        # )

        # combined = torch.cat([
        #     protein_feat,
        #     motif_output[:, 0, :]
        # ], dim=1)

        # 门控机制酶特征融合
        fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat([protein_esm2, protein_struct], -1)))
        enzyme_feat = fusion_gate * protein_esm2 + (1 - fusion_gate) * protein_struct

        # 酶到底物的注意力
        enzyme_as_query = self.enzyme_attn(
            query=enzyme_feat.unsqueeze(0),
            key=atom_gl_feat.unsqueeze(0),
            value=atom_gl_feat.unsqueeze(0)
        )[0].squeeze(0)

        # 底物到酶的注意力
        substrate_as_query = self.substrate_attn(
            query=atom_gl_feat.unsqueeze(0),
            key=enzyme_feat.unsqueeze(0),
            value=enzyme_feat.unsqueeze(0)
        )[0].squeeze(0)

        # 对称交互
        combined = torch.cat([
            enzyme_feat + 0.3 * enzyme_as_query,  # 酶特征+底物上下文
            atom_gl_feat + 0.3 * substrate_as_query  # 底物特征+酶上下文
        ], dim=-1)

        return self.classifier(self.fusion(combined))
