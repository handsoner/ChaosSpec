"""
inter_fusion.py

This module implements the ChaosSpec framework: an ultra-lightweight, single-source
domain adaptive multimodal disease diagnosis model based on the fusion of Raman
and Infrared spectroscopy, enhanced by chaotic features.

Author: Wu Mingtao
Date: 2025/12/23
"""

import torch
import torch.nn.functional as F
from torch import nn


class MultimodalModel(nn.Module):
    """
    Main architecture of the ChaosSpec Framework.
    Integrates Chaotic Enhanced Encoders, Bi-Spectral Cross-Modal Interaction, and Prediction modules.
    """
    def __init__(self, input_dim1, input_dim2, output_dim, dropout_rate=0.2, hidden_dims=44):
        super(MultimodalModel, self).__init__()

        # Component 1: Intra-Modal Attentional Encoders (for Infrared and Raman respectively)
        self.ir_intra = EncoderWithIntra(input_dim1, hidden_dims, output_intra=20, rank=20, dropout=dropout_rate)
        self.raman_intra = EncoderWithIntra(input_dim2, hidden_dims, output_intra=20, rank=20, dropout=dropout_rate)

        # Component 2: Bi-Spectral Cross-Modal Interaction
        # enhanced_dim = hidden_dims + output_intra = 44 + 20 = 64
        self.cross_interaction = CrossModalInteraction(enhanced_dim=64, output_inter=20, rank=20, dropout=dropout_rate)

        # Component 3: Sequential Prediction Module
        # in_size = (hidden_dims + output_intra) + output_inter = 64 + 20 = 84
        self.predictor = Predictor(in_size=84, hidden_size=hidden_dims, output_dim=output_dim, dropout_rate=dropout_rate)

    def forward(self, infrared, raman, infrared_chao, raman_chao):
        # 1. Concatenate high-dimensional spectral data with low-dimensional chaotic features
        infrared_combined = torch.cat((infrared, infrared_chao), dim=1)
        raman_combined = torch.cat((raman, raman_chao), dim=1)

        # 2. Intra-modal feature extraction and attention enhancement
        feat_ir, attn_ir = self.ir_intra(infrared_combined)
        feat_ra, attn_ra = self.raman_intra(raman_combined)

        # Unimodal summation (Residual-like connection)
        unimodal_sum = feat_ir + feat_ra

        # 3. Bi-spectral cross-modal interaction
        bimodal_feat = self.cross_interaction(feat_ir, feat_ra, attn_ir, attn_ra)

        # 4. Final prediction
        out = self.predictor(unimodal_sum, bimodal_feat)
        return out


class EncoderWithIntra(nn.Module):
    """
    Intra-Modal Attentional Encoder.
    Extracts unimodal representations and applies a self-pooling attention mechanism
    to filter noise and amplify pathology-relevant spectral envelopes.
    """
    def __init__(self, input_dim, hidden_dim, output_intra, rank, dropout=0.0):
        super(EncoderWithIntra, self).__init__()

        # 1. Base feature extraction pipeline
        encoder1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        encoder2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.encoder = nn.Sequential(encoder1, encoder2)

        # 2. Intra-modal projection (for Low-rank Bilinear Pooling / MFB)
        self.linear_intra = nn.Linear(hidden_dim, rank * output_intra)

        # 3. Intra-modal attention gate
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + output_intra, 1),
            nn.Sigmoid()
        )

        self.rank = rank
        self.output_intra = output_intra
        self.drop = nn.Dropout(dropout)

    def mfb_self_pooling(self, x, out_dim):
        """
        Multimodal Factorized Bilinear (MFB) self-pooling to capture
        second-order interactions within the same modality.
        """
        fusion = torch.mul(x, x)
        fusion = self.drop(fusion)
        fusion = fusion.view(-1, 1, out_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        # Signed square root to preserve magnitude direction
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        return F.normalize(fusion)

    def forward(self, x):
        # Extract abstract base features
        base_feat = self.encoder(x.squeeze(1))

        # Project and apply intra-modal self-pooling
        intra_proj = self.linear_intra(base_feat)
        intra_feat = self.mfb_self_pooling(intra_proj, self.output_intra)

        # Feature concatenation and attention scoring
        combined = torch.cat((base_feat, intra_feat), 1)
        attn_score = self.attention(combined)

        # Adaptive feature enhancement
        enhanced_feat = attn_score * combined

        return enhanced_feat, attn_score


class CrossModalInteraction(nn.Module):
    """
    Bi-Spectral Cross-Modal Interaction module.
    Explores biochemical complementarity between Infrared and Raman signatures
    using low-rank bilinear pooling and global alignment soft-gating.
    """
    def __init__(self, enhanced_dim, output_inter, rank, dropout=0.0):
        super(CrossModalInteraction, self).__init__()
        self.rank = rank
        self.output_inter = output_inter
        self.drop = nn.Dropout(dropout)

        # Cross-modal projections
        self.linear_inter_1 = nn.Linear(enhanced_dim, rank * output_inter)
        self.linear_inter_2 = nn.Linear(enhanced_dim, rank * output_inter)

    def mfb_cross_pooling(self, x1, x2, out_dim):
        """
        Computes Hadamard-product-based interaction between two modalities.
        """
        fusion = torch.mul(x1, x2)
        fusion = self.drop(fusion)
        fusion = fusion.view(-1, 1, out_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        return F.normalize(fusion)

    def forward(self, feat1, feat2, attn1, attn2):
        # 1. Calculate cross-modal global alignment weights (sgp)
        g_norm = F.softmax(feat1, 1)
        p_norm = F.softmax(feat2, 1)

        # Inner product to capture global semantic correlations
        dot_prod = torch.matmul(g_norm.unsqueeze(1), p_norm.unsqueeze(2)).squeeze()
        sgp = (1 / (dot_prod + 0.5) * (attn1.squeeze() + attn2.squeeze()))

        # Normalize the soft-gating weight
        sgp_a = F.softmax(sgp.unsqueeze(1), 1)[:, 0].unsqueeze(1).expand(-1, self.output_inter)

        # 2. Execute cross-modal nonlinear coupling (Hadamard product interaction)
        proj1 = self.linear_inter_1(feat1)
        proj2 = self.linear_inter_2(feat2)
        inter_feat = self.mfb_cross_pooling(proj1, proj2, self.output_inter)

        # 3. Generate final bi-spectral fused representation
        bimodal_feat = sgp_a * inter_feat
        return bimodal_feat


class Predictor(nn.Module):
    """
    Sequential Prediction module.
    Utilizes Bi-LSTM to capture cross-band temporal dependencies and dynamic feature evolution.
    """
    def __init__(self, in_size, hidden_size, output_dim, dropout_rate=0.0):
        super(Predictor, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)

        # Deep sequential feature extraction
        self.lstm = nn.LSTM(in_size, 64, num_layers=1, batch_first=True)
        self.norm1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)

        # Final classification mapping
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, unimodal_sum, bimodal_feat):
        # Fuse unimodal aggregates and bi-spectral interaction features
        fusion = torch.cat((unimodal_sum, bimodal_feat), 1)
        fusion = self.norm(fusion)

        # Sequence modeling for context evolution
        code, (final_hidden_state, final_cell_state) = self.lstm(fusion)

        # Normalization, regularization, and classification
        code = self.norm1(code)
        code = self.dropout(code)
        out = self.classifier(code)

        return out