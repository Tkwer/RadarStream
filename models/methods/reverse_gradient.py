import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    

# MultiView 对抗性多视角对齐  
class DomainAdversarialNetwork(nn.Module):  
    def __init__(self, feature_dim, num_domains, method='standard', num_views=1):  
        super().__init__()  
        self.method = method
        self.num_views = num_views
        if self.method == 'DScombine':
            input_size = feature_dim * num_views
        else:
            input_size = feature_dim
        # 域判别器  
        self.domain_classifier = nn.Sequential(  
            nn.Linear(input_size, input_size // 2),  
            nn.ReLU(),  
            nn.Linear(input_size // 2, num_domains),  
            nn.Softmax(dim=-1)  
        )  


    def forward(self, features, alpha):  
        # 如果是DScombine方法，合并特征
        if self.method == 'DScombine':
            features = torch.cat(features, dim=-1) # 合并所有视角的特征
          # 通过反向梯度层
        reverse_features = ReverseLayerF.apply(features, alpha)

        # 域分类
        domain_logits = self.domain_classifier(reverse_features)
        return domain_logits