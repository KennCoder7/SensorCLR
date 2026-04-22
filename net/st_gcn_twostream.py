import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_ft = kwargs['return_ft']
        kwargs['return_ft'] = True
        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)
        self.fusion = nn.Sequential(nn.Linear(256, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5))
        self.cls = nn.Linear(512, kwargs['num_class'])

    def forward(self, x):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        C = 3
        x = x[:, :C, :, :, :]
        V = 25
        x = x[:, :, :, :V, :]
        motion = x[:, :, 1::, :, :] - x[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        m = F.upsample(motion, size=(T, V), mode='bilinear',
                       align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        _, feats_x = self.origin_stream(x)
        _, feats_m = self.motion_stream(m)
        res = torch.cat((feats_x, feats_m), dim=1)
        res = self.fusion(res)
        cls = self.cls(res)
        if self.return_ft:
            return cls, res
        else:
            return cls
