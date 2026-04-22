import math
import random
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Backbone definitions
# =========================
class CNN(nn.Module):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=9):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            self._make_layers(input_channel, hidden_base, (6, 1), (3, 1), (1, 0)),
            self._make_layers(hidden_base, hidden_base * 2, (6, 1), (3, 1), (1, 0)),
            self._make_layers(hidden_base * 2, hidden_base * 4, (6, 1), (3, 1), (1, 0)),
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.hidden_dim = hidden_base * 4
        self.num_axis = num_axis
        self.fc = nn.Linear(self.hidden_dim * num_axis, num_classes)

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
                return_sequence=False):
        # x: [B, T, D]
        x = x.unsqueeze(1)
        if freeze_encoder:
            with torch.no_grad():
                feat_map = self.encoder(x)
        else:
            feat_map = self.encoder(x)

        # feat_map: [B, C, T', A]
        B, C, Tp, A = feat_map.shape
        seq_feat = feat_map.mean(dim=3).transpose(1, 2).contiguous()  # [B, T', C]

        if return_sequence:
            return seq_feat

        global_feat = feat_map.mean(dim=2).reshape(B, C * A)

        if only_encoder:
            return global_feat

        out = self.fc(global_feat)
        if with_feat:
            return out, global_feat
        return out



class CNN_OPPO(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_OPPO, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.encoder = nn.Sequential(
            self._make_layers(input_channel, hidden_base, 3, 2, 1),
            self._make_layers(hidden_base, hidden_base * 2, 3, 2, 1),
            self._make_layers(hidden_base * 2, hidden_base * 4, 3, 2, 1),
        )
        self.hidden_dim = hidden_base * 4
        self.num_axis = 7
        self.fc = nn.Linear(self.hidden_dim * self.num_axis, num_classes)


class CNN_UniMiB(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_UniMiB, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.hidden_dim = hidden_base * 4
        self.num_axis = 3
        self.fc = nn.Linear(self.hidden_dim * self.num_axis, num_classes)


class CNN_PAMAP2(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_PAMAP2, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.hidden_dim = hidden_base * 4
        self.num_axis = 36
        self.fc = nn.Linear(self.hidden_dim * self.num_axis, num_classes)


class CNN_WISDM(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=3):
        super(CNN_WISDM, self).__init__(num_classes, input_channel, freeze, hidden_base, num_axis)


BACKBONE_ZOO = {
    'CNN': CNN,
    'CNN_OPPO': CNN_OPPO,
    'CNN_UniMiB': CNN_UniMiB,
    'CNN_PAMAP2': CNN_PAMAP2,
    'CNN_WISDM': CNN_WISDM,
}


def build_backbone(backbone_name='CNN', num_classes=60, hidden_base=64, freeze=False, **kwargs):
    if backbone_name not in BACKBONE_ZOO:
        raise ValueError(f"Unknown backbone_name={backbone_name}. Available: {list(BACKBONE_ZOO.keys())}")
    backbone_cls = BACKBONE_ZOO[backbone_name]
    return backbone_cls(num_classes=num_classes, hidden_base=hidden_base, freeze=freeze, **kwargs)



# =========================
# AimTS utilities
# =========================
class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims=128, hidden_dims=256):
        super().__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, output_dims)
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.repr_dropout(self.proj_head(x))


class Jittering(nn.Module):
    def __init__(self, sigma=0.3):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        ret = x + torch.normal(mean=0.0, std=self.sigma, size=x.shape, device=x.device)
        return torch.nan_to_num(ret)


class Scaling(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        factor = torch.normal(mean=1.0, std=self.sigma, size=(x.shape[0], x.shape[2]), device=x.device)
        ret = x * factor.unsqueeze(1)
        return torch.nan_to_num(ret)


class WindowSlicing(nn.Module):
    def __init__(self, reduce_ratio=0.5):
        super().__init__()
        self.reduce_ratio = reduce_ratio

    def forward(self, x):
        B, T, D = x.shape
        target_len = max(2, int(math.ceil(self.reduce_ratio * T)))
        if target_len >= T:
            return x
        starts = torch.randint(low=0, high=T - target_len + 1, size=(B,), device=x.device)
        pieces = []
        for i in range(B):
            seg = x[i:i + 1, starts[i]:starts[i] + target_len].transpose(1, 2)
            seg = F.interpolate(seg, size=T, mode='linear', align_corners=False).transpose(1, 2)
            pieces.append(seg)
        return torch.cat(pieces, dim=0)


class WindowWarping(nn.Module):
    def __init__(self, window_ratio=0.3, scales=(0.5, 2.0)):
        super().__init__()
        self.window_ratio = window_ratio
        self.scales = scales

    def forward(self, x):
        B, T, D = x.shape
        warp_size = max(2, int(math.ceil(self.window_ratio * T)))
        if T - warp_size - 1 <= 1:
            return x
        starts = torch.randint(low=1, high=T - warp_size, size=(B,), device=x.device)
        rets = []
        for i in range(B):
            scale = random.choice(self.scales)
            seg = x[i:i + 1, starts[i]:starts[i] + warp_size].transpose(1, 2)
            seg = F.interpolate(seg, size=max(2, int(warp_size * scale)), mode='linear', align_corners=False)
            left = x[i:i + 1, :starts[i]]
            right = x[i:i + 1, starts[i] + warp_size:]
            merged = torch.cat([left, seg.transpose(1, 2), right], dim=1).transpose(1, 2)
            merged = F.interpolate(merged, size=T, mode='linear', align_corners=False).transpose(1, 2)
            rets.append(merged)
        return torch.cat(rets, dim=0)


class TimeWarping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, T, D = x.shape
        grid = torch.linspace(-1, 1, T, device=x.device)
        noise = 0.08 * torch.sin(torch.linspace(0, math.pi * 2, T, device=x.device))[None, :, None]
        warped = grid[None, :, None] + noise
        warped = warped.clamp(-1, 1)
        inp = x.transpose(1, 2).unsqueeze(-1)
        sample_grid = torch.zeros(B, T, 1, 2, device=x.device)
        sample_grid[..., 0] = 0
        sample_grid[..., 1] = warped.squeeze(-1)
        out = F.grid_sample(inp, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out.squeeze(-1).transpose(1, 2)


def _instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def contrastive_loss(z1, z2):
    loss = z1.new_tensor(0.0)
    depth = 0
    while z1.size(1) > 1:
        loss = loss + _instance_contrastive_loss(z1, z2)
        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        loss = loss + _instance_contrastive_loss(z1, z2)
        depth += 1
    return loss / max(depth, 1)


def compute_tao(ts_list1: List[torch.Tensor], ts_list2: List[torch.Tensor]):
    G = len(ts_list1)
    distance_matrix = torch.zeros((2 * G, 2 * G), dtype=torch.float32, device=ts_list1[0].device)

    def _dis(a, b):
        distances = torch.norm(a - b, dim=2)
        return torch.mean(distances / a.shape[1]).item()

    for i in range(G):
        sum_i = 0.0
        sum_iG = 0.0
        for j in range(G):
            distance_matrix[i, j] = _dis(ts_list1[i], ts_list1[j])
            distance_matrix[i + G, j] = _dis(ts_list2[i], ts_list1[j])
            distance_matrix[i, j + G] = _dis(ts_list1[i], ts_list2[j])
            distance_matrix[i + G, j + G] = _dis(ts_list2[i], ts_list2[j])
            sum_i += distance_matrix[i, j] + distance_matrix[i, j + G]
            sum_iG += distance_matrix[i + G, j] + distance_matrix[i + G, j + G]
        distance_matrix[i, i] = sum_i / max(2 * G - 1, 1)
        distance_matrix[i + G, i + G] = sum_iG / max(2 * G - 1, 1)

    for i in range(2):
        for j in range(2):
            sub = distance_matrix[i * G:(i + 1) * G, j * G:(j + 1) * G].clone()
            sub[torch.eye(G, dtype=torch.bool, device=sub.device)] = -float('inf')
            distance_matrix[i * G:(i + 1) * G, j * G:(j + 1) * G] = sub

    matrix = F.softmax(distance_matrix, dim=1) + 1.0
    a, b = torch.min(matrix).item(), torch.max(matrix).item()
    if abs(b - a) < 1e-12:
        return torch.ones_like(matrix) * 1.005
    return 1 + (matrix - a) * (0.01 / (b - a))


def proto_loss(z1, z2, tao_matrix, proj_head):
    # z1,z2: [G,B,T,C]
    G, B, T, C = z1.shape
    z1p = proj_head(z1.reshape(G * B * T, C)).reshape(G, B, T, -1)
    z2p = proj_head(z2.reshape(G * B * T, C)).reshape(G, B, T, -1)
    centroids1 = z1p.mean(dim=0)
    centroids2 = z2p.mean(dim=0)
    loss_inter = contrastive_loss(centroids1, centroids2)

    D = z1p.shape[-1]
    z1g = z1p.permute(0, 1, 3, 2).reshape(G * B * D, T)
    z2g = z2p.permute(0, 1, 3, 2).reshape(G * B * D, T)
    z1g = F.max_pool1d(z1g.unsqueeze(1), kernel_size=T).squeeze(1).reshape(G, B, D)
    z2g = F.max_pool1d(z2g.unsqueeze(1), kernel_size=T).squeeze(1).reshape(G, B, D)

    if G == 1:
        return loss_inter, z1.new_tensor(0.0)

    weight = torch.eye(G, device=z1.device)
    mat0 = torch.tril(weight, diagonal=-1)[:, :-1]
    mat0 += torch.triu(weight, diagonal=1)[:, 1:]
    labels_L = torch.cat([mat0, weight], dim=1)
    labels_R = torch.cat([weight, mat0], dim=1)

    z = torch.cat([z1g, z2g], dim=0).transpose(0, 1)  # [B,2G,D]
    sim = torch.matmul(z, z.transpose(1, 2))
    sim = sim / tao_matrix.unsqueeze(0)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(G, device=z1.device)
    loss_intra = torch.sum(logits[:, i] * labels_L) + torch.sum(logits[:, G + i] * labels_R)
    loss_intra /= (4 * G * B)
    return loss_inter, loss_intra


def sensor_align_loss(z1, z2, scale=20.0):
    """Bidirectional alignment loss between two sensor views."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = scale * (z1 @ z2.t())
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# =========================
# Main model
# =========================
class AimTS(nn.Module):
    """
    Self-contained AimTS-style model for sensor-only SSL.

    Inference:
        out, feat = model(x)

    Pretraining:
        ret = model(batch_or_x, im_k=True)
        loss = ret['loss']

    Supported pretraining input:
        1) Tensor [B,T,D]
        2) dict {'data': Tensor [B,T,D]}  (timestamp channel is NOT expected here)
    """
    def __init__(self,
                 backbone_name='CNN',
                 num_classes=6,
                 input_dims=9,
                 hidden_base=64,
                 feature_dim=128,
                 warmup_epochs=1,
                 align_weight=1.0,
                 proto_weight=1.0,
                 weak_aug='jitter_scale',
                 num_axis=None,
                 freeze=False,
                 backbone_kwargs=None,
                 **kwargs):
        super().__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.align_weight = align_weight
        self.proto_weight = proto_weight
        self.weak_aug = weak_aug

        backbone_kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        self.encoder_q = build_backbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            hidden_base=hidden_base,
            freeze=freeze,
            **backbone_kwargs,
        )
        dim_mlp = self.encoder_q.fc.weight.shape[1]

        self.proj_head_ts = ProjectionHead(dim_mlp, output_dims=feature_dim, hidden_dims=256)
        seq_dim = self.encoder_q.hidden_dim if hasattr(self.encoder_q, 'hidden_dim') else hidden_base * 4
        self.proj_head_pro = ProjectionHead(seq_dim, output_dims=feature_dim, hidden_dims=256)

        self.augs = nn.ModuleList([
            Jittering(), Scaling(), WindowSlicing(), TimeWarping(), WindowWarping()
        ])
        self.weak_augs = nn.ModuleList([
            Jittering(sigma=0.1), Scaling(sigma=0.2)
        ])

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _apply_weak_aug(self, x):
        if self.weak_aug == 'none':
            return x
        out = x
        for aug in self.weak_augs:
            out = aug(out)
        return out

    def _encode_sequence(self, x):
        seq = self.encoder_q(x, return_sequence=True)     # [B,T',C]
        feat = self.encoder_q(x, only_encoder=True)       # [B,Cg]
        return seq, feat

    def forward(self, im_q, im_k=None, freeze_encoder=True):
        # inference mode
        if im_k is None:
            out, feat = self.encoder_q(im_q, with_feat=True, freeze_encoder=freeze_encoder)
            feat = F.normalize(feat, dim=1)
            return out, feat

        # pretraining mode
        if isinstance(im_q, dict):
            if 'data' not in im_q:
                raise ValueError("AimTS pretraining dict must contain key 'data'")
            x = im_q['data']
        else:
            x = im_q

        if x.shape[-1] != self.input_dims:
            raise ValueError(f"input_dims mismatch: model expects {self.input_dims}, but batch has {x.shape[-1]}")

        # 1) sensor-sensor alignment branch (replace original ts-image alignment)
        xw1 = self._apply_weak_aug(x.clone())
        xw2 = self._apply_weak_aug(x.clone())
        _, feat1 = self._encode_sequence(xw1)
        _, feat2 = self._encode_sequence(xw2)
        emb1 = self.proj_head_ts(feat1)
        emb2 = self.proj_head_ts(feat2)
        loss_align = sensor_align_loss(emb1, emb2)

        # 2) prototype branch on multi-augmentation sequence features
        loss_proto = emb1.new_tensor(0.0)
        loss_inter = emb1.new_tensor(0.0)
        loss_intra = emb1.new_tensor(0.0)

        if self.current_epoch >= self.warmup_epochs:
            aug1_all = [aug(x.clone()) for aug in self.augs]
            aug2_all = [aug(x.clone()) for aug in self.augs]
            out1_list, out2_list = [], []
            ts_list1, ts_list2 = [], []

            for x1, x2 in zip(aug1_all, aug2_all):
                ts_list1.append(x1)
                ts_list2.append(x2)
                seq1, _ = self._encode_sequence(x1)
                seq2, _ = self._encode_sequence(x2)
                out1_list.append(seq1)
                out2_list.append(seq2)

            tao_aug = compute_tao(ts_list1, ts_list2)
            out1 = torch.stack(out1_list, dim=0)  # [G,B,T',C]
            out2 = torch.stack(out2_list, dim=0)
            loss_inter, loss_intra = proto_loss(out1, out2, tao_aug, self.proj_head_pro)
            loss_proto = 0.9 * loss_inter + 0.1 * loss_intra

        loss = self.align_weight * loss_align + self.proto_weight * loss_proto
        return {
            'loss': loss,
            'loss_align': loss_align,
            'loss_proto': loss_proto,
            'loss_inter': loss_inter,
            'loss_intra': loss_intra,
            'emb1': emb1,
            'emb2': emb2,
        }
