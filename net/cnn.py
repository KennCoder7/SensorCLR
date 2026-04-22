import torch
import torch.nn as nn


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

        self.fc = nn.Linear(hidden_base * 4 * 4 * num_axis, num_classes)
        # self.fc = nn.Linear(hidden_base * 4, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, only_encoder=False, with_feat=False, freeze_encoder=False):
        # x = self.data_batch_norm(x)
        x = x.unsqueeze(1)
        if freeze_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        # print('x.shape', x.shape)
        x = x.view(x.size(0), -1)
        if only_encoder:
            return x
        # out = self.flatten(x)
        out = self.fc(x)
        # out = F.softmax(out)
        if with_feat:
            return out, x
        else:
            return out


class CNN_OPPO(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_OPPO, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.encoder = nn.Sequential(
            self._make_layers(input_channel, hidden_base, 3, 2, 1),
            self._make_layers(hidden_base, hidden_base * 2, 3, 2, 1),
            self._make_layers(hidden_base * 2, hidden_base * 4, 3, 2, 1),
        )
        self.fc = nn.Linear(hidden_base * 4 * 8 * 7, num_classes)


class CNN_UniMiB(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_UniMiB, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.fc = nn.Linear(hidden_base * 4 * 5 * 3, num_classes)

class CNN_PAMAP2(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64):
        super(CNN_PAMAP2, self).__init__(num_classes, input_channel, freeze, hidden_base)
        self.fc = nn.Linear(hidden_base * 4 * 5 * 36, num_classes)

class CNN_WISDM(CNN):
    def __init__(self, num_classes, input_channel=1, freeze=False, hidden_base=64, num_axis=3):
        super(CNN_WISDM, self).__init__(num_classes, input_channel, freeze, hidden_base, num_axis)
