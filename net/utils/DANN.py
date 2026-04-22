import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 两个类别：源域和目标域
        )

    def forward(self, x):
        return self.fc(x)


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.lambda_val
        grad_input = grad_output.neg() * lambda_val
        return grad_input, None
