import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DualMoCo(nn.Module):
    def __init__(self, base_encoder, pretrain=True, num_classes=60,
                 momentum=0.999, Temperature=0.07, queue_size=32768,
                 mlp=True, feature_dim=256, nnm_topk=1, hidden_base=64, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        self.pretrain = pretrain

        self.K = queue_size
        self.m = momentum
        self.T = Temperature
        self.nnm_topk = nnm_topk

        base_encoder = import_class(base_encoder)
        self.encoder_q = base_encoder(num_classes=num_classes, hidden_base=hidden_base)
        self.encoder_k = base_encoder(num_classes=num_classes, hidden_base=hidden_base)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.fc_q = MLPHead(dim_mlp, out_dim=feature_dim)
            self.fc_k = MLPHead(dim_mlp, out_dim=feature_dim)
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue1", torch.randn(feature_dim, queue_size))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.register_buffer("queue2", torch.randn(feature_dim, queue_size))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if hasattr(self, 'fc_q') and hasattr(self, 'fc_k'):
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        batch_size = keys1.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys1.device.index
        self.queue1[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys1.T
        self.queue2[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys2.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, nnm=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        # print('im_q shape:', im_q.shape)
        # exit()

        if im_k is None:
            out, feat = self.encoder_q(im_q, with_feat=True, freeze_encoder=True)
            feat = F.normalize(feat, dim=1)
            return out, feat

        # compute query features
        q1 = self.encoder_q(im_q, only_encoder=True)  # queries: NxC
        q1 = self.fc_q(q1) if hasattr(self, 'fc_q') else q1
        q1 = F.normalize(q1, dim=1)

        q2 = self.encoder_q(im_k, only_encoder=True)  # queries: NxC
        q2 = self.fc_q(q2) if hasattr(self, 'fc_q') else q2
        q2 = F.normalize(q2, dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1 = self.encoder_k(im_k, only_encoder=True)  # keys: NxC
            k1 = self.fc_k(k1) if hasattr(self, 'fc_k') else k1
            k1 = F.normalize(k1, dim=1)

            k2 = self.encoder_k(im_q, only_encoder=True)  # keys: NxC
            k2 = self.fc_k(k2) if hasattr(self, 'fc_k') else k2
            k2 = F.normalize(k2, dim=1)

        # compute logits
        l_pos1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1)

        l_neg1 = torch.einsum('nc,ck->nk', [q1, self.queue1.clone().detach()])
        l_neg2 = torch.einsum('nc,ck->nk', [q2, self.queue2.clone().detach()])

        logits1 = torch.cat([l_pos1, l_neg1], dim=1) / self.T
        logits2 = torch.cat([l_pos2, l_neg2], dim=1) / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()


        self._dequeue_and_enqueue(k1, k2)

        if nnm:
            l_neg = (l_neg1 + l_neg2) / 2
            _, topkdix = torch.topk(l_neg, self.nnm_topk, dim=1)

            topk_onehot = torch.zeros_like(l_neg)
            topk_onehot.scatter_(1, topkdix, 1)

            pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

            return logits1, logits2, pos_mask

        return logits1, logits2, labels

class SensorCLR(nn.Module):
    def __init__(self, base_encoder, pretrain=True, num_classes=60,
                 momentum=0.999, Temperature=0.07, queue_size=32768,
                 mlp=True, feature_dim=256, nnm_topk=1, hidden_base=64,
                 **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        self.pretrain = pretrain

        self.K = queue_size
        self.m = momentum
        self.T = Temperature
        self.nnm_topk = nnm_topk

        base_encoder = import_class(base_encoder)
        self.encoder_q = base_encoder(num_classes=num_classes, hidden_base=hidden_base)
        self.encoder_k = base_encoder(num_classes=num_classes, hidden_base=hidden_base)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.fc_q = MLPHead(dim_mlp, out_dim=feature_dim)
            self.fc_k = MLPHead(dim_mlp, out_dim=feature_dim)
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue1", torch.randn(feature_dim, queue_size))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.register_buffer("queue2", torch.randn(feature_dim, queue_size))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if hasattr(self, 'fc_q') and hasattr(self, 'fc_k'):
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        batch_size = keys1.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys1.device.index
        self.queue1[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys1.T
        self.queue2[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys2.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, im_e1=None, im_e2=None, nnm=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if im_k is None:
            out, feat = self.encoder_q(im_q, with_feat=True, freeze_encoder=True)
            feat = F.normalize(feat, dim=1)
            return out, feat

        # compute query features
        q1 = self.encoder_q(im_q, only_encoder=True)  # queries: NxC
        q1 = self.fc_q(q1) if hasattr(self, 'fc_q') else q1
        q1 = F.normalize(q1, dim=1)

        q2 = self.encoder_q(im_k, only_encoder=True)  # queries: NxC
        q2 = self.fc_q(q2) if hasattr(self, 'fc_q') else q2
        q2 = F.normalize(q2, dim=1)

        # compute extra query features
        q_e1 = self.encoder_q(im_e1, only_encoder=True)  # queries: NxC
        q_e1 = self.fc_q(q_e1) if hasattr(self, 'fc_q') else q_e1
        q_e1 = F.normalize(q_e1, dim=1)

        q_e2 = self.encoder_q(im_e2, only_encoder=True)  # queries: NxC
        q_e2 = self.fc_q(q_e2) if hasattr(self, 'fc_q') else q_e2
        q_e2 = F.normalize(q_e2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1 = self.encoder_k(im_k, only_encoder=True)  # keys: NxC
            k1 = self.fc_k(k1) if hasattr(self, 'fc_k') else k1
            k1 = F.normalize(k1, dim=1)

            k2 = self.encoder_k(im_q, only_encoder=True)  # keys: NxC
            k2 = self.fc_k(k2) if hasattr(self, 'fc_k') else k2
            k2 = F.normalize(k2, dim=1)

        # compute logits
        l_pos1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1)

        l_neg1 = torch.einsum('nc,ck->nk', [q1, self.queue1.clone().detach()])
        l_neg2 = torch.einsum('nc,ck->nk', [q2, self.queue2.clone().detach()])

        logits1 = torch.cat([l_pos1, l_neg1], dim=1) / self.T
        logits2 = torch.cat([l_pos2, l_neg2], dim=1) / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()

        l_pos_e1 = torch.einsum('nc,nc->n', [q_e1, k1]).unsqueeze(-1)
        l_pos_e2 = torch.einsum('nc,nc->n', [q_e2, k2]).unsqueeze(-1)
        l_neg_e1 = torch.einsum('nc,ck->nk', [q_e1, self.queue1.clone().detach()])
        l_neg_e2 = torch.einsum('nc,ck->nk', [q_e2, self.queue2.clone().detach()])
        logits_e1 = torch.cat([l_pos_e1, l_neg_e1], dim=1) / self.T
        logits_e2 = torch.cat([l_pos_e2, l_neg_e2], dim=1) / self.T

        self._dequeue_and_enqueue(k1, k2)

        if nnm:
            l_neg = (l_neg1 + l_neg2) / 2
            l_neg_e = (l_neg_e1 + l_neg_e2) / 2
            _, topkdix = torch.topk(l_neg, self.nnm_topk, dim=1)
            _, topkdix_e = torch.topk(l_neg_e, self.nnm_topk, dim=1)

            topk_onehot = torch.zeros_like(l_neg)
            topk_onehot.scatter_(1, topkdix, 1)
            topk_onehot.scatter_(1, topkdix_e, 1)

            pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

            return logits1, logits2, pos_mask, logits_e1, logits_e2

        return logits1, logits2, labels, logits_e1, logits_e2

    def __init__(self, base_encoder, pretrain=True, num_classes=60,
                 momentum=0.999, Temperature=0.07, queue_size=32768,
                 mlp=True, feature_dim=256, nnm_topk=1, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        self.pretrain = pretrain

        self.K = queue_size
        self.m = momentum
        self.T = Temperature
        self.nnm_topk = nnm_topk

        base_encoder = import_class(base_encoder)
        self.encoder_q = base_encoder(num_classes=num_classes)
        self.encoder_k = base_encoder(num_classes=num_classes)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.fc_q = MLPHead(dim_mlp, out_dim=feature_dim)
            self.fc_k = MLPHead(dim_mlp, out_dim=feature_dim)
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue1", torch.randn(feature_dim, queue_size))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.register_buffer("queue2", torch.randn(feature_dim, queue_size))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))