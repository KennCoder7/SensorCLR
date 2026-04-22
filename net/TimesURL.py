import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Backbone definitions
# =========================
class CNN(nn.Module):
    """
    Default CNN backbone for sensor sequences.

    Input:
        x: [B, T, D]  (time length T, sensor dimension D)

    Supports:
        - return_sequence=True  -> [B, T', C] for TimesURL
        - only_encoder=True     -> [B, C*A]  for global feature extraction
        - with_feat=True        -> (logits, feat)
    """
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
        self.fc = nn.Linear(self.hidden_dim * self.num_axis, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, only_encoder=False, with_feat=False, freeze_encoder=False, return_sequence=False):
        x = x.unsqueeze(1)  # [B, 1, T, D]
        if freeze_encoder:
            with torch.no_grad():
                feat_map = self.encoder(x)
        else:
            feat_map = self.encoder(x)

        # feat_map: [B, C, T', A]
        B, C, Tp, A = feat_map.shape

        # sequence feature for TimesURL
        seq_feat = feat_map.mean(dim=3).transpose(1, 2).contiguous()  # [B, T', C]
        if return_sequence:
            return seq_feat

        # global feature for classifier / MoCo / DINO style use
        global_feat = feat_map.mean(dim=2).reshape(B, C * A)          # [B, C*A]
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
# Heads
# =========================
class ReconstructionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# TimesURL utilities
# =========================
def take_per_row(x, starts, length):
    device = x.device
    B = x.size(0)
    idx = starts[:, None] + torch.arange(length, device=device)[None, :]
    if x.dim() in (2, 3):
        return x[torch.arange(B, device=device)[:, None], idx]
    raise ValueError(f"Unsupported x.dim={x.dim()}")


def instance_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha, beta = 0.2, 0.2
    if B == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[torch.randperm(B, device=z1.device), :, :].view_as(z1)
    uni_z2 = beta * z2 + (1 - beta) * z2[torch.randperm(B, device=z2.device), :, :].view_as(z2)

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)  # [4B, T, C]
    z = z.transpose(0, 1)                            # [T, 4B, C]
    sim = torch.matmul(z[:, :2 * B, :], z.transpose(1, 2)) / temp

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits[:, :2 * B, :(2 * B - 1)]

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha, beta = 0.2, 0.2
    if T == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[:, torch.randperm(T, device=z1.device), :].view_as(z1)
    uni_z2 = beta * z2 + (1 - beta) * z2[:, torch.randperm(T, device=z2.device), :].view_as(z2)

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=1)  # [B, 4T, C]
    sim = torch.matmul(z[:, :2 * T, :], z.transpose(1, 2)) / temp

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits[:, :2 * T, :(2 * T - 1)]

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.8, temporal_unit=0, temp=1.0):
    loss = z1.new_tensor(0.)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss_mixup(z1, z2, temp)
        if d >= temporal_unit and (1 - alpha) != 0:
            loss = loss + (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1 and alpha != 0:
        loss = loss + alpha * instance_contrastive_loss_mixup(z1, z2, temp)
        d += 1
    return loss / max(d, 1)


def freq_mix(x, rate=0.5, dim=1):
    x_f = torch.fft.fft(x, dim=dim)
    m = torch.rand(x_f.shape, device=x.device) < rate
    amp = torch.abs(x_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)

    freal = x_f.real.masked_fill(m, 0)
    fimag = x_f.imag.masked_fill(m, 0)

    b_idx = torch.randperm(x.shape[0], device=x.device)
    x2_f = torch.fft.fft(x[b_idx], dim=dim)

    m2 = torch.bitwise_not(m)
    freal = freal + x2_f.real.masked_fill(m2, 0)
    fimag = fimag + x2_f.imag.masked_fill(m2, 0)

    x_mix = torch.fft.ifft(torch.complex(freal, fimag), dim=dim)
    return torch.abs(x_mix)


def tp_noneffect(func, x, time_dim=-1, **kwargs):
    tp = x[..., time_dim:time_dim + 1]
    value = x[..., :time_dim]
    value = func(value, **kwargs)
    return torch.cat([value, tp], dim=-1)


# =========================
# TimesURL main model
# =========================
class TimesURL(nn.Module):
    """
    All-in-one TimesURL with selectable built-in CNN backbone.

    Args:
        backbone_name: one of ['CNN', 'CNN_OPPO', 'CNN_UniMiB', 'CNN_PAMAP2', 'CNN_WISDM']
        num_classes: for inference classifier head dimension in backbone
        input_dims: number of value channels D for reconstruction target

    Training input format:
        batch = {
            'data': [B, T, D+1],   # last channel is timestamp
            'mask': [B, T, D],
            'mask_inter': [B, T, D]   # optional
        }
    """
    def __init__(self,
                 backbone_name='CNN',
                 pretrain=True,
                 num_classes=60,
                 input_dims=None,
                 hidden_base=64,
                 feature_dim=256,
                 proj_hidden_dim=512,
                 recon_hidden_dim=512,
                 temporal_unit=0,
                 alpha=0.8,
                 temperature=1.0,
                 recon_weight=1.0,
                 contrast_weight=1.0,
                 freeze=False,
                 backbone_kwargs=None,
                 **kwargs):
        super().__init__()
        self.pretrain = pretrain
        self.temporal_unit = temporal_unit
        self.alpha = alpha
        self.temp = temperature
        self.recon_weight = recon_weight
        self.contrast_weight = contrast_weight

        backbone_kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        self.encoder_q = build_backbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            hidden_base=hidden_base,
            freeze=freeze,
            **backbone_kwargs,
        )

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.proj = ProjectionHead(self.encoder_q.hidden_dim, hidden_dim=proj_hidden_dim, out_dim=feature_dim)

        # reconstruction predicts original value channels, not timestamp
        if input_dims is None:
            input_dims = num_classes
        self.input_dims = input_dims
        self.recon_head = ReconstructionHead(self.encoder_q.hidden_dim, recon_hidden_dim, out_dim=input_dims)

    def _crop_views(self, x, ts_l):
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        crop_offset = torch.as_tensor(crop_offset, device=x.device, dtype=torch.long)
        return crop_l, crop_left, crop_right, crop_eleft, crop_eright, crop_offset

    def _encode_sequence(self, x):
        seq = self.encoder_q(x, return_sequence=True)
        if seq.dim() != 3:
            raise ValueError(f"TimesURL requires [B,T,C] sequence features, got {tuple(seq.shape)}")
        z = self.proj(seq)
        z = F.normalize(z, dim=-1)
        return seq, z

    def forward(self, im_q, im_k=None, freeze_encoder=True):
        # inference mode: keep MoCo-like style
        if im_k is None:
            out, feat = self.encoder_q(im_q, with_feat=True, freeze_encoder=freeze_encoder)
            feat = F.normalize(feat, dim=1)
            return out, feat

        if not isinstance(im_q, dict):
            raise ValueError("For TimesURL pretraining, im_q should be a dict with keys: data, mask, mask_inter")

        x = im_q['data']
        mask = im_q['mask']
        mask_inter = im_q.get('mask_inter', None)

        # print(x.shape)
        # exit()
        B, T, Dp1 = x.shape
        D = Dp1 - 1
        if D != self.input_dims:
            raise ValueError(f"input_dims mismatch: model expects {self.input_dims}, but batch has {D}")

        crop_l, crop_left, crop_right, crop_eleft, crop_eright, crop_offset = self._crop_views(x, T)
        # safer minimum temporal length for CNN backbones
        min_crop_len = 64
        crop_l = max(int(crop_l), min_crop_len)
        crop_l = min(crop_l, T)

        x_left = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        x_right_raw = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        x_right = tp_noneffect(freq_mix, x_right_raw, time_dim=D, rate=0.5)

        mask1 = take_per_row(mask, crop_offset + crop_eleft, crop_right - crop_eleft)
        mask2 = take_per_row(mask, crop_offset + crop_left, crop_eright - crop_left)

        if mask_inter is not None:
            mask1_inter = take_per_row(mask_inter, crop_offset + crop_eleft, crop_right - crop_eleft)
            mask2_inter = take_per_row(mask_inter, crop_offset + crop_left, crop_eright - crop_left)
        else:
            mask1_inter = None
            mask2_inter = None

        # actual available temporal lengths after crop
        left_len = x_left.shape[1]
        right_len = x_right.shape[1]

        # final crop length cannot exceed the shorter branch
        crop_l = min(crop_l, left_len, right_len)

        # if still too short, pad temporally to avoid CNN kernel crash
        if crop_l < min_crop_len:
            pad_len_left = max(0, min_crop_len - left_len)
            pad_len_right = max(0, min_crop_len - right_len)

            if pad_len_left > 0:
                x_left = F.pad(x_left, (0, 0, pad_len_left, 0), mode='replicate')
                mask1 = F.pad(mask1, (0, 0, pad_len_left, 0), mode='replicate')
                if mask1_inter is not None:
                    mask1_inter = F.pad(mask1_inter, (0, 0, pad_len_left, 0), mode='replicate')

            if pad_len_right > 0:
                x_right = F.pad(x_right, (0, 0, 0, pad_len_right), mode='replicate')
                mask2 = F.pad(mask2, (0, 0, 0, pad_len_right), mode='replicate')
                if mask2_inter is not None:
                    mask2_inter = F.pad(mask2_inter, (0, 0, 0, pad_len_right), mode='replicate')

            crop_l = min_crop_len

        x_left = x_left[:, -crop_l:]
        x_right = x_right[:, :crop_l]
        mask1 = mask1[:, -crop_l:]
        mask2 = mask2[:, :crop_l]
        if mask1_inter is not None:
            mask1_inter = mask1_inter[:, -crop_l:]
            mask2_inter = mask2_inter[:, :crop_l]

        # masked input for encoder
        x_left_in = torch.cat([x_left[..., :D] * mask1, x_left[..., D:]], dim=-1)
        x_right_in = torch.cat([x_right[..., :D] * mask2, x_right[..., D:]], dim=-1)

        seq1, z1 = self._encode_sequence(x_left_in)   # seq1: [B, T1, C]
        seq2, z2 = self._encode_sequence(x_right_in)  # seq2: [B, T2, C]

        loss_cl = hierarchical_contrastive_loss(
            z1, z2, alpha=self.alpha, temporal_unit=self.temporal_unit, temp=self.temp
        )

        loss_recon = z1.new_tensor(0.)
        if mask1_inter is not None:
            rec1 = self.recon_head(seq1)   # expected [B, T1, D]
            rec2 = self.recon_head(seq2)   # expected [B, T2, D]

            target1 = x_left[..., :D]       # [B, crop_l, D]
            target2 = x_right[..., :D]      # [B, crop_l, D]

            # align temporal length by interpolation if needed
            if rec1.shape[1] != target1.shape[1]:
                rec1 = F.interpolate(
                    rec1.transpose(1, 2),   # [B, D, T1]
                    size=target1.shape[1],  # -> crop_l
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)           # [B, crop_l, D]

            if rec2.shape[1] != target2.shape[1]:
                rec2 = F.interpolate(
                    rec2.transpose(1, 2),   # [B, D, T2]
                    size=target2.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)           # [B, crop_l, D]

            # final safety check
            if rec1.shape != target1.shape:
                raise ValueError(f"rec1 shape mismatch after interpolation: rec1={rec1.shape}, target1={target1.shape}")
            if rec2.shape != target2.shape:
                raise ValueError(f"rec2 shape mismatch after interpolation: rec2={rec2.shape}, target2={target2.shape}")
            if mask1_inter.shape != target1.shape:
                raise ValueError(f"mask1_inter shape mismatch: mask1_inter={mask1_inter.shape}, target1={target1.shape}")
            if mask2_inter.shape != target2.shape:
                raise ValueError(f"mask2_inter shape mismatch: mask2_inter={mask2_inter.shape}, target2={target2.shape}")

            denom1 = mask1_inter.sum() + 1e-10
            denom2 = mask2_inter.sum() + 1e-10

            if mask1_inter.sum() > 0:
                loss_recon = loss_recon + (((target1 - rec1) * mask1_inter) ** 2).sum() / denom1 / 2
            if mask2_inter.sum() > 0:
                loss_recon = loss_recon + (((target2 - rec2) * mask2_inter) ** 2).sum() / denom2 / 2

        loss = self.contrast_weight * loss_cl + self.recon_weight * loss_recon
        return {
            'loss': loss,
            'loss_cl': loss_cl,
            'loss_recon': loss_recon,
            'z1': z1,
            'z2': z2,
        }