# -*- coding: utf-8 -*-
import torch
import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import torch.nn as nn
from torch.nn import functional as F
from thop import profile
THOP_AVAILABLE = True
import time
def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix

def regularize_weights(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

def CoxLoss1(survtime, censor, hazard_pred,device):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def accuracy_cox(hazardsdata, labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def CIndex_lifeline(hazards, labels, survtime_all):
    return concordance_index(survtime_all, -hazards, labels)

def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]: concord = concord + 1
                    elif hazards[j] < hazards[i]: concord = concord + 0.5

    return(concord/total)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def CoxLoss2(survtime, censor, hazard_pred,device):
    n_observed = censor.sum(0)+1
    ytime_indicator = R_set(survtime)
    ytime_indicator = torch.FloatTensor(ytime_indicator).to(device)
    risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred))
    diff = hazard_pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1))
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class CosineSimilarity(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return torch.sum(F.cosine_similarity(x1, x2, self.dim, self.eps))/x1.size(0)
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        # print("pred sizeis : ", pred.size())
        # print("real sizeis : ", real.size())
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n-torch.sum(diffs.pow(2)) / (n*n)

        return mse


def calculate_model_efficiency(model, data, device):
    """
    独立计算模型的 GFLOPs 和 Inference Time
    """
    print("\n" + "=" * 20 + " Model Efficiency Calculation " + "=" * 20)

    # 1. 准备数据：复用 DatasetLoader 获取一个 batch 的正确格式输入
    from data_loaders import graph_fusion_DatasetLoader
    from torch.utils.data import DataLoader

    # 创建一个临时的 loader 只为了获取输入维度
    temp_loader = DataLoader(dataset=graph_fusion_DatasetLoader(data, split='test'),
                             batch_size=2, shuffle=False)

    model.eval()

    # 获取第一个 Batch
    for x_gene, x_path, x_gene_chao, x_path_chao, label in temp_loader:
        # 展平数据 (与 train/test 中的预处理保持一致)
        x_gene = x_gene.view(x_gene.size(0), -1).to(device)
        x_path = x_path.view(x_path.size(0), -1).to(device)
        x_gene_chao = x_gene_chao.view(x_gene_chao.size(0), -1).to(device)
        x_path_chao = x_path_chao.view(x_path_chao.size(0), -1).to(device)

        inputs = (x_gene, x_path, x_gene_chao, x_path_chao)
        break  # 只需要一个样本

    # -------------------------------------------------------
    # 2. 计算 GFLOPs (使用 thop)
    # -------------------------------------------------------
    if THOP_AVAILABLE:
        try:
            # profile 会自动挂钩子计算
            flops, params = profile(model, inputs=inputs, verbose=False)
            print(f"[Complexity] Params: {params / 1e6 :.4f} M")
            print(f"[Complexity] FLOPs:  {flops / 1e9 :.4f} G")
        except Exception as e:
            print(f"[Complexity] Error calculating FLOPs: {e}")
    else:
        print("[Complexity] 'thop' library not found. Install via 'pip install thop'")

    # -------------------------------------------------------
    # 3. 计算 Inference Time (纯推理耗时，排除数据加载)
    # -------------------------------------------------------
    print("[Timing] Measuring inference time (100 loops)...")

    # 预热 (Warm up)
    with torch.no_grad():
        for _ in range(10):
            _ = model(*inputs)

    # 开始计时
    iterations = 100

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(*inputs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)  # 返回毫秒
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(*inputs)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000  # 转换为毫秒

    avg_time_ms = elapsed_time_ms / iterations
    fps = 1000 / avg_time_ms

    print(f"[Timing] Average Inference Time: {avg_time_ms:.4f} ms/sample")
    print(f"[Timing] FPS (Throughput):       {fps:.2f} samples/sec")
    print("=" * 64 + "\n")