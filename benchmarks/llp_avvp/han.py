# Adapted from AVVP-ECCV20 (Tian, Li & Xu, "Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video
# Parsing", ECCV 2020), the LLP benchmark authors' official code: the Hybrid Attention Network (HAN) with attentive
# multimodal multiple-instance-learning pooling (MMIL_Net), their baseline, which they train on frozen pre-extracted
# features (VGGish audio, ResNet-152 frames, R(2+1)D-18 snippets). It is this folder's readout head.
#   repository  https://github.com/YapengTian/AVVP-ECCV20
#   commit      fde56118f2c2e7c19188dfefa2569a37b70054fe
#   path        nets/net_audiovisual.py
#   licence     GNU General Public License v3.0, as the repository README states (the repository ships no LICENSE
#               file). This file keeps that licence (https://www.gnu.org/licenses/gpl-3.0).
#   changes     (each marked "yavih:" below) MMIL_Net takes the features' width ``dim``: fc_a and fc_v map ``dim`` to 512
#               instead of 128 (VGGish) and 2048 (ResNet-152); fc_st and fc_fusion (the R(2+1)D branch and the 2D/3D
#               fusion) are removed and forward() takes one per-second visual feature, so x2 = fc_v(visual) replaces
#               the fusion. Everything else (Encoder, HANLayer, CMTLayer, the unused audio_encoder, visual_encoder
#               and cmt_encoder modules the upstream model also builds, the attentive MMIL pooling) is unchanged.
#               The per-second pooling of the bridge's tokens and the training loss are in avvp.py.
# Linting is off for the copy (it stays as upstream):
# ruff: noqa
# ----------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v

class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)



class MMIL_Net(nn.Module):

    def __init__(self, dim):  # yavih: the frozen features' width ``dim`` (upstream: no argument)
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(dim, 512)  # yavih: Linear(128, 512) upstream (VGGish)
        self.fc_v = nn.Linear(dim, 512)  # yavih: Linear(2048, 512) upstream (ResNet-152); fc_st, fc_fusion removed
        self.audio_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.visual_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.cmt_encoder = Encoder(CMTLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.hat_encoder = Encoder(HANLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)

    def forward(self, audio, visual):  # yavih: audio, visual [B, 10, dim], one pooled feature per second (no visual_st)

        x1 = self.fc_a(audio)

        # yavih: upstream fuses 2D and 3D visual features here (fc_v on 8 frames per second, avg_pool2d over them, fc_st
        # on the R(2+1)D feature, concatenation, fc_fusion); the bridge's per-second visual feature takes their place
        x2 = self.fc_v(visual)

        # HAN
        x1, x2 = self.hat_encoder(x1, x2)

        # prediction
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob*av_att).sum(dim=2).sum(dim=1)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob =temporal_prob[:, :, 1, :].sum(dim=1)

        return global_prob, a_prob, v_prob, frame_prob

class CMTLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CMTLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout1(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q
