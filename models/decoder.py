import torch
import torch.nn as nn
from models.methods.attention import (
    MultiViewLinearProjectionAttention,
    MultiViewSEAttention,
    MultiViewECAAttention,
    AdaptiveMultiViewAttention,
    MultiViewDSFusion
)

# 使用工厂字典映射不同的attention方法
ATTENTION_FACTORIES = {
    'linear_projection': lambda num_views, feature_dim, is_sharedspecific: MultiViewLinearProjectionAttention(num_views, feature_dim, is_sharedspecific),  
    'se_attention': lambda num_views, feature_dim, is_sharedspecific: MultiViewSEAttention(num_views, feature_dim, is_sharedspecific),  
    'eca_attention': lambda num_views, feature_dim, is_sharedspecific: MultiViewECAAttention(num_views, feature_dim, is_sharedspecific),  
    'adaptive_attention': lambda num_views, feature_dim, is_sharedspecific: AdaptiveMultiViewAttention(num_views, feature_dim, is_sharedspecific),  
}

class MultiviewDecoder(nn.Module):
    """
    Base class for multi-view decoders, supports different fusion modes.
    """
    def __init__(self, feature_dim, num_views=None, mode='concat'):
        super(MultiviewDecoder, self).__init__()
        self.mode = mode
        self.num_views = num_views
        self.feature_dim = feature_dim
        # 为每个输入特征创建可学习的权重  
 
    def add_fusion(self, *x_list):
        # Dynamically sum features along the last dimension.
        # 确保 x_list 是一个包含张量的列表  
        if not x_list:  
            return None  # 如果没有输入，返回 None 或者可以抛出异常  

        # 使用 torch.stack 将所有输入张量沿着新维度堆叠  
        stacked_tensors = torch.stack(x_list, dim=0)  # 形状为 (num_views, batch_size, feature_dim)  
        
        # 对最后一个维度进行求和  
        return torch.sum(stacked_tensors, dim=0)  # 返回形状为 (batch_size, feature_dim)  

    def concat_fusion(self, *x_list):
        # Dynamically concatenate features along the last dimension.
        return torch.cat(x_list, dim=-1)

    def attention_fusion(self, *x_list, target_domains=None):
        # Use attention module for feature fusion
        views = list(x_list)
        return self.attention(views)

    def forward(self, *x_list, **kwargs):
        if self.num_views is not None and len(x_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} inputs, got {len(x_list)}")

        if self.mode == 'add':
            return self.add_fusion(*x_list)
        elif self.mode == 'concat':
            return self.concat_fusion(*x_list)
        elif self.mode == 'attention':
            return self.attention_fusion(*x_list, **kwargs)
        else:
            raise ValueError("Invalid fusion mode")

# add 是concat的特殊形式
class ConcatDecoder(MultiviewDecoder):  
    """  
    Simply concatenates input features from multiple views.  
    """  
    def __init__(self, feature_dim, num_views=None, mode='concat'):  
        super().__init__(feature_dim=feature_dim, num_views=num_views, mode=mode)  
        # 计算输入特征的大小  
        if mode=='concat':
            input_size = feature_dim * num_views  # 假设每个视图的特征维度均为 feature_dim  
        elif mode=='add':
            input_size = feature_dim  # 假设每个视图的特征维度均为 feature_dim 

        # 添加线性层，输入大小为 input_size，输出大小为 feature_dim  
        self.linear = nn.Linear(input_size, feature_dim)  
        
    def forward(self, *x_list, **kwargs):  
        # 调用父类的 forward 方法  
        fused_features = super().forward(*x_list, **kwargs)  
        
        # 通过线性层输出  
        output = self.linear(fused_features)  
        
        return output, None 

class AttentionDecoder(MultiviewDecoder):  
    """  
    Aligns multiple views using different attention strategies.  
    """  
    def __init__(self, feature_dim, num_views=None, is_sharedspecific=0, method='linear_projection', num_classes=7):  
        super().__init__(feature_dim=feature_dim, num_views=num_views, mode='attention')  

        self.method = method  
        if method=='DScombine':
            self.attention = MultiViewDSFusion(num_views, feature_dim, num_classes)
        else:
            if method not in ATTENTION_FACTORIES:  
                raise ValueError(f"Unknown attention type: {method}")  
            self.attention = ATTENTION_FACTORIES[method](num_views, feature_dim, is_sharedspecific)  

    def forward(self, *x_list, **kwargs):  
        # 直接调用父类的 forward 方法  
        return super().forward(*x_list, **kwargs)  

