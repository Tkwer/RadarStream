import torch
import torch.nn.functional as F

class DirichletCombiner:
    def __init__(self, classes):
        self.classes = classes

    def DS_Combin_two(self, alpha1, alpha2):
        """
        组合两个视图的Dirichlet分布参数。

        :param alpha1: 视图1的Dirichlet参数 (Tensor of shape [batch_size, classes])
        :param alpha2: 视图2的Dirichlet参数 (Tensor of shape [batch_size, classes])
        :return: 合并后的Dirichlet参数 (Tensor of shape [batch_size, classes])
        """
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)  # 每个类的总和
            E[v] = alpha[v] - 1  # E = alpha - 1
            b[v] = E[v] / S[v].expand_as(E[v])  # b = E / S
            u[v] = self.classes / S[v]  # u = classes / S

        # 计算 b^0 @ b^1
        bb = torch.bmm(b[0].unsqueeze(2), b[1].unsqueeze(1))  # [batch_size, classes, classes]
        # 计算 C
        bb_sum = torch.sum(bb, dim=(1, 2))  # [batch_size]
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)  # [batch_size]
        C = bb_sum - bb_diag  # [batch_size]

        # 计算 b^a
        bu = b[0] * u[1].expand_as(b[0])
        ub = b[1] * u[0].expand_as(b[1])
        numerator_b = (b[0] * b[1]) + bu + ub  # [batch_size, classes]
        denominator = (1 - C).unsqueeze(1).expand_as(numerator_b) + 1e-10  # 防止除零
        b_a = numerator_b / denominator  # [batch_size, classes]

        # 计算 u^a
        u_a = (u[0] * u[1]) / (1 - C).unsqueeze(1).expand_as(b[0])  # [batch_size, classes]

        # 计算新的 S
        S_a = self.classes / u_a  # [batch_size, classes]

        # 计算新的 e_k
        e_a = b_a * S_a  # [batch_size, classes]
        alpha_a = e_a + 1  # [batch_size, classes]

        return alpha_a, u_a, u[0], u[1]

    def DS_Combin(self, alphas):
        """
        使用链式法则组合多个视图的Dirichlet分布参数。

        :param alphas: 包含所有视图的Dirichlet参数列表 (List of Tensors, each of shape [batch_size, classes])
        :return: 合并后的Dirichlet参数 (Tensor of shape [batch_size, classes]) u_a是融合后的不确定度，u_list是每个视图的不确定度
        """
        if not alphas:
            raise ValueError("The 'alphas' list must contain at least one Dirichlet parameter tensor.")

        # 初始化为第一个视图的参数
        alpha_combined = alphas[0]
        u_list =[]
        # 逐步组合剩余视图
        for i in range(1, len(alphas)):
            alpha_combined, u_a, u_prev, u_new = self.DS_Combin_two(alpha_combined, alphas[i])
            u_list.append(u_prev)
            if i == (len(alphas)-1):
                u_list.append(u_new)
        u_tensor = torch.cat(u_list, dim=-1)        
        return alpha_combined, u_a, u_tensor

def KL_divergence(alpha, c, device='cuda'):
    """
    计算Dirichlet分布的KL散度与均匀Dirichlet分布。

    :param alpha: Dirichlet参数 (Tensor of shape [batch_size, classes])
    :param c: 类别数
    :param device: 设备（'cuda' 或 'cpu'）
    :return: KL散度 (Tensor of shape [batch_size, 1])
    """
    beta = torch.ones((1, c), device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)  # [batch_size, 1]
    S_beta = torch.sum(beta, dim=1, keepdim=True)    # [1, 1]
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)  # [batch_size, 1]
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)  # [1, 1]
    dg0 = torch.digamma(S_alpha)  # [batch_size, 1]
    dg1 = torch.digamma(alpha)     # [batch_size, classes]
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni  # [batch_size, 1]
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step, device='cuda'):
    """
    计算交叉熵损失，包括Dirichlet分布的KL散度。

    :param p: 真实标签 (Tensor of shape [batch_size])
    :param alpha: Dirichlet参数 (Tensor of shape [batch_size, classes])
    :param c: 类别数
    :param global_step: 当前训练步数
    :param annealing_step: 进行退火的步数
    :param device: 设备（'cuda' 或 'cpu'）
    :return: 交叉熵损失 (Tensor of shape [batch_size, 1])
    """
    S = torch.sum(alpha, dim=1, keepdim=True)  # [batch_size, 1]
    E = alpha - 1  # [batch_size, classes]
    label = F.one_hot(p, num_classes=c).float()  # [batch_size, classes]
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)  # [batch_size, 1]

    annealing_coef = min(1.0, global_step / annealing_step)  # scalar

    alp = E * (1 - label) + 1  # [batch_size, classes]
    B = annealing_coef * KL_divergence(alp, c, device=device)  # [batch_size, 1]

    return A + B  # [batch_size, 1]

def combined_loss(p, alphas, c, alpha_combined, global_step, annealing_step, device='cuda'):
    """
    计算多视图的总损失，包括每个视图的损失和融合后的总损失。

    :param p: 真实标签 (Tensor of shape [batch_size])
    :param alphas: 包含所有视图的Dirichlet参数列表 (List of Tensors, each of shape [batch_size, classes])
    :param c: 类别数
    :param alpha_combined: 融合后的证据
    :param global_step: 当前训练步数
    :param annealing_step: 进行退火的步数
    :param device: 设备（'cuda' 或 'cpu'）
    :return: 总损失 (Tensor of shape [batch_size])
    """
    if not alphas:
        raise ValueError("The 'alphas' list must contain at least one Dirichlet parameter tensor.")

    # 计算每个视图的损失
    per_view_losses = []
    for i in range(len(alphas)):
        loss = ce_loss(p, alphas[i], c, global_step, annealing_step, device=device)  # [batch_size, 1]
        per_view_losses.append(loss)

    # 计算融合后的损失
    fused_loss = ce_loss(p, alpha_combined, c, global_step, annealing_step, device=device)  # [batch_size, 1]

    # 将所有损失堆叠起来并求和
    total_loss = torch.cat(per_view_losses + [fused_loss], dim=1).sum(dim=1)  # [batch_size]

    return total_loss.mean(0)  # 可以根据需要求平均

# 示例用法
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 5
    classes = 3
    combiner = DirichletCombiner(classes=classes)

    # 假设有三个视图，每个视图的alpha参数为[batch_size, classes]的Tensor
    alpha0 = torch.tensor([[1.2, 3.4, 2.2],
                           [2.1, 1.9, 3.0],
                           [1.5, 2.5, 2.5],
                           [3.0, 1.0, 2.0],
                           [2.2, 2.2, 1.6]], device=device).float()

    alpha1 = torch.tensor([[2.3, 1.7, 3.3],
                           [1.8, 2.2, 2.0],
                           [2.0, 2.0, 2.0],
                           [1.5, 2.5, 3.0],
                           [3.0, 1.5, 1.5]], device=device).float()

    alpha2 = torch.tensor([[1.0, 2.0, 3.0],
                           [2.0, 1.0, 3.0],
                           [3.0, 2.0, 1.0],
                           [1.5, 1.5, 2.0],
                           [2.5, 2.5, 0.5]], device=device).float()

    alphas = [alpha0, alpha1, alpha2]
    p = torch.tensor([0, 1, 2, 1, 0], device=device)  # 真实标签

    global_step = 50
    annealing_step = 100

    loss = combined_loss(p, alphas, classes, combiner, global_step, annealing_step, device=device)
    print("Combined Loss:\n", loss)
