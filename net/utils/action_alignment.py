import torch
import torch.nn.functional as F
import numpy as np


def action_alignment(A, B, method='mean', view_weight=None, norm=True):
    """
    输入:
    - A: (nA, views, dim) 的张量，代表A中实体的特征
    - B: (nB, views, dim) 的张量，代表B中实体的特征
    - method: 聚合方式，默认为 'mean'，也可以选择 'sum'

    输出:
    - similarity_matrix: A中每个实体与B中每个实体的相似度矩阵
    """

    nA, viewsA, dimA = A.shape
    nB, viewsB, dimB = B.shape

    # 保证A和B有相同的views数
    assert viewsA == viewsB, "A and B must have the same number of views"

    if view_weight is None:
        view_weight = np.ones(viewsA)

    # 初始化相似度矩阵
    similarity_matrix = torch.zeros((nA, nB)).to(A.device)

    # 对每个视角进行相似度计算
    for v in range(viewsA):
        # 获取第v个视角的特征
        A_view = A[:, v, :]  # A 中第 v 个视角的特征 (nA, dim)
        B_view = B[:, v, :]  # B 中第 v 个视角的特征 (nB, dim)

        # 计算当前视角下A与B的相似度
        # similarity = torch.matmul(A_view, B_view.T)  # (nA, nB)
        similarity = F.cosine_similarity(A_view.unsqueeze(1), B_view.unsqueeze(0), dim=2)  # (nA, nB)

        # 将相似度累加到总相似度矩阵中
        similarity_matrix += view_weight[v] * similarity

    # 根据选择的聚合方法对视角进行聚合
    if method == 'mean':
        similarity_matrix /= viewsA  # 对所有视角求平均
    elif method == 'sum':
        pass  # 已经累加过，无需再操作

    # 选择Softmax归一化
    # similarity_matrix = F.softmax(similarity_matrix, dim=1)

    # 0-1归一化
    if norm:
        similarity_matrix = normalize_similarity_per_line(similarity_matrix)

    return similarity_matrix


def normalize_similarity(similarity_matrix):
    # 计算矩阵的最小值和最大值
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()

    # Min-Max归一化
    normalized_similarity = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized_similarity


def normalize_similarity_per_line(similarity_matrix):
    # 计算矩阵的最小值和最大值
    min_val = similarity_matrix.min(dim=1, keepdim=True)[0]
    max_val = similarity_matrix.max(dim=1, keepdim=True)[0]

    # Min-Max归一化
    normalized_similarity = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized_similarity


def index_alignment(similarity_matrix):
    similarity_matrix = torch.tensor(similarity_matrix).clone().detach()
    A2B = torch.argmax(similarity_matrix, dim=1)
    B2A = torch.argmax(similarity_matrix, dim=0)
    return (A2B, B2A, 'source') if len(A2B) < len(B2A) else (A2B, B2A, 'target')
