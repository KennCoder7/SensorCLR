import math
from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Backbones
# =========================================================
class CNN(nn.Module):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=9):
        super(CNN, self).__init__()

        self.num_axis = num_axis
        self.hidden_dim = hidden_base * 4

        self.encoder = nn.Sequential(
            self._make_layers(input_channel, hidden_base, (6, 1), (3, 1), (1, 0)),
            self._make_layers(hidden_base, hidden_base * 2, (6, 1), (3, 1), (1, 0)),
            self._make_layers(hidden_base * 2, hidden_base * 4, (6, 1), (3, 1), (1, 0)),
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(hidden_base * 4 * num_axis, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self,
                x,
                only_encoder=False,
                with_feat=False,
                freeze_encoder=False,
                return_sequence=False,
                hierarchical=False):
        # x: [B, T, D]
        x = x.unsqueeze(1)  # [B,1,T,D]

        feats = []
        out = x

        layers = list(self.encoder)
        if freeze_encoder:
            with torch.no_grad():
                for layer in layers:
                    out = layer(out)
                    feats.append(out)
        else:
            for layer in layers:
                out = layer(out)
                feats.append(out)

        if hierarchical:
            return feats  # list of [B,C,T',D]

        feat_map = out  # [B,C,T',A]
        B, C, Tp, A = feat_map.shape

        seq_feat = feat_map.mean(dim=3).transpose(1, 2).contiguous()  # [B,T',C]
        if return_sequence:
            return seq_feat

        global_feat = feat_map.mean(dim=2).reshape(B, C * A)  # [B,C*A]

        if only_encoder:
            return global_feat

        # print(f"Global feat shape: {global_feat.shape}")
        out = self.fc(global_feat)
        if with_feat:
            return out, global_feat
        return out


class CNN_OPPO(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=7):
        super().__init__(num_classes, input_channel, freeze, hidden_base, num_axis)
        self.num_axis = 7
        self.hidden_dim = hidden_base * 4
        self.encoder = nn.Sequential(
            self._make_layers(input_channel, hidden_base, 3, 2, 1),
            self._make_layers(hidden_base, hidden_base * 2, 3, 2, 1),
            self._make_layers(hidden_base * 2, hidden_base * 4, 3, 2, 1),
        )
        self.fc = nn.Linear(64 * 4 * 7, num_classes)


class CNN_UniMiB(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=3):
        super().__init__(num_classes, input_channel, freeze, hidden_base, num_axis)


class CNN_PAMAP2(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=36):
        super().__init__(num_classes, input_channel, freeze, hidden_base, num_axis)


class CNN_WISDM(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=3):
        super().__init__(num_classes, input_channel, freeze, hidden_base, num_axis)


def build_backbone(backbone_name: str, num_classes: int, hidden_base: int, input_dims: int):
    name = backbone_name.lower()
    if name == 'cnn':
        return CNN(num_classes=num_classes, hidden_base=hidden_base, num_axis=input_dims)
    if name == 'cnn_oppo':
        return CNN_OPPO(num_classes=num_classes, hidden_base=hidden_base, num_axis=input_dims)
    if name == 'cnn_unimib':
        return CNN_UniMiB(num_classes=num_classes, hidden_base=hidden_base, num_axis=input_dims)
    if name == 'cnn_pamap2':
        return CNN_PAMAP2(num_classes=num_classes, hidden_base=hidden_base, num_axis=input_dims)
    if name == 'cnn_wisdm':
        return CNN_WISDM(num_classes=num_classes, hidden_base=hidden_base, num_axis=input_dims)
    raise ValueError(f'Unknown backbone_name: {backbone_name}')


# =========================================================
# Hierarchical encoder wrapper
# =========================================================
class ConvEncoderHier(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.enc_feat_map_chs = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x, hierarchical=True)
        if self.enc_feat_map_chs is None:
            self.enc_feat_map_chs = [f.shape[1] for f in feats]
        return feats

    @staticmethod
    def get_downsample_ratio_from_feats(x: torch.Tensor, feats: List[torch.Tensor]) -> int:
        # x: [B,T,D], feats[-1]: [B,C,T',A]
        t_in = x.shape[1]
        t_out = feats[-1].shape[2]
        return max(1, math.ceil(t_in / max(1, t_out)))


# =========================================================
# Decoder
# =========================================================
class SimpleTemporalDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, feats: torch.Tensor, target_t: int = None):
        # feats: [B,T',C]
        rec = self.proj(feats)  # [B,T',D]
        if target_t is not None and rec.shape[1] != target_t:
            rec = F.interpolate(
                rec.transpose(1, 2),
                size=target_t,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        return rec


# =========================================================
# MaskCAE
# =========================================================
class MaskCAE(nn.Module):
    def __init__(self,
                 backbone_name='CNN_OPPO',
                 num_classes=6,
                 input_dims=77,
                 hidden_base=64,
                 feature_dim=128,
                 mask_ratio=0.6,
                 patch_size=4,
                 recon_weight=1.0):
        super().__init__()

        self.input_dims = input_dims
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.recon_weight = recon_weight

        self.encoder_q = build_backbone(backbone_name, num_classes, hidden_base, input_dims)
        self.encoder_hier = ConvEncoderHier(self.encoder_q)

        self.encoder_out_dim = self.encoder_q.hidden_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.encoder_out_dim, self.encoder_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_out_dim, feature_dim)
        )
        self.decoder = SimpleTemporalDecoder(self.encoder_out_dim, input_dims)

    def _make_patch_mask(self, B, T, D, device):
        """
        Return:
            visible_mask: [B,T,D], bool
            True means visible, False means masked
        """
        num_patch = math.ceil(T / self.patch_size)

        visible_patch = (torch.rand(B, num_patch, device=device) > self.mask_ratio)  # [B,P]
        visible = visible_patch.unsqueeze(-1).expand(B, num_patch, self.patch_size)   # [B,P,ps]
        visible = visible.reshape(B, num_patch * self.patch_size)[:, :T]               # [B,T]

        if visible.dim() == 2:
            visible = visible.unsqueeze(-1)  # [B,T,1]

        return visible.expand(B, T, D)  # [B,T,D]

    def _align_rec(self, rec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        rec:    [B,T',D']
        target: [B,T,D]
        """
        B, T_target, D_target = target.shape

        if rec.dim() != 3:
            raise ValueError(f"Unexpected rec shape: {rec.shape}")

        if rec.shape[1] != T_target:
            rec = F.interpolate(
                rec.transpose(1, 2),
                size=T_target,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        if rec.shape[2] != D_target:
            if rec.shape[2] > D_target:
                rec = rec[..., :D_target]
            else:
                rec = F.pad(rec, (0, D_target - rec.shape[2]))

        if rec.shape != (B, T_target, D_target):
            raise ValueError(f"rec shape mismatch: rec={rec.shape}, target={(B, T_target, D_target)}")
        return rec

    def _encode_masked_sequence(self, x_masked: torch.Tensor):
        """
        x_masked: [B,T,D]
        Returns:
            seq_feat:  [B,T',C]
            global_z:  [B,F]
            feats_hier: list of feat maps
            downsample_ratio: dynamic ratio
        """
        feats_hier = self.encoder_hier(x_masked)
        downsample_ratio = self.encoder_hier.get_downsample_ratio_from_feats(x_masked, feats_hier)

        last_feat = feats_hier[-1]  # [B,C,T',A]
        seq_feat = last_feat.mean(dim=3).transpose(1, 2).contiguous()  # [B,T',C]
        global_feat = seq_feat.mean(dim=1)  # [B,C]
        z = self.proj_head(global_feat)
        z = F.normalize(z, dim=1)

        return seq_feat, z, feats_hier, downsample_ratio

    def forward(self, im_q: Union[torch.Tensor, Dict[str, torch.Tensor]], im_k=None, freeze_encoder=True):
        # downstream inference/classification
        if im_k is None:
            if isinstance(im_q, dict):
                x = im_q['data']
            else:
                x = im_q
            out, feat = self.encoder_q(x, with_feat=True, freeze_encoder=freeze_encoder)
            feat = F.normalize(feat, dim=1)
            return out, feat

        # pretraining
        if isinstance(im_q, dict):
            x = im_q['data']
        else:
            x = im_q

        B, T, D = x.shape
        if D != self.input_dims:
            raise ValueError(f"input_dims mismatch: model expects {self.input_dims}, but batch has {D}")

        visible = self._make_patch_mask(B, T, D, x.device)   # [B,T,D], bool
        visible_f = visible.float()
        masked_f = 1.0 - visible_f

        x_masked = x * visible_f

        seq_feat, z, feats_hier, downsample_ratio = self._encode_masked_sequence(x_masked)

        rec = self.decoder(seq_feat, target_t=T)
        rec = self._align_rec(rec, x)

        denom = masked_f.sum() + 1e-10
        loss_recon = (((x - rec) * masked_f) ** 2).sum() / denom

        return {
            'loss': self.recon_weight * loss_recon,
            'loss_recon': loss_recon,
            'z': z,
            'rec': rec,
            'visible_mask': visible,
            'downsample_ratio': downsample_ratio,
        }