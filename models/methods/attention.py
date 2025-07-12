import torch
import torch.nn as nn
import torch.nn.functional as F
from models.methods.DScombine import DirichletCombiner 

# Basic Shared Model
class BasicSharedSpecificModel(nn.Module):
    def __init__(self, feature_dim, num_views):
        super(BasicSharedSpecificModel, self).__init__()
        self.shared_layer = nn.Linear(num_views * feature_dim, feature_dim)
        self.specific_layer = nn.Linear(num_views * feature_dim, feature_dim)
        self.fusion_layer = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, input):
        shared_output = self.shared_layer(input)
        specific_output = self.specific_layer(input)
        output = self.fusion_layer(torch.cat([shared_output, specific_output], dim=-1))
        return output

# Base MultiView Attention Class
class MultiViewAttention(nn.Module):
    def __init__(self, num_views, feature_dim, is_sharedspecific):
        super().__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        if is_sharedspecific:
            self.fusion = BasicSharedSpecificModel(feature_dim, num_views)
        else:
            self.fusion = nn.Linear(feature_dim * num_views, feature_dim) 

    def weight_generator(self, views):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, views):
        view_weights = self.weight_generator(views)  # Generate view weights (batch_size, num_views)
        views_tensor = torch.stack(views, dim=-1)  # (batch_size, feature_dim, num_views)
        aligned_views = views_tensor * view_weights.unsqueeze(1)  # (batch_size, feature_dim, num_views)

        cross_view =  aligned_views.view(aligned_views.shape[0], -1)  # 形状 (batch_size, feature_dim * num_views)    # (batch_size, feature_dim * num_views)
        alignment = self.fusion(cross_view)  # (batch_size, feature_dim)
        return alignment, view_weights

# Linear Projection Attention
class MultiViewLinearProjectionAttention(MultiViewAttention):
    def __init__(self, num_views, feature_dim, is_sharedspecific):
        super().__init__(num_views, feature_dim, is_sharedspecific)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Softmax(dim=-1)
            ) for _ in range(num_views)
        ])

    def weight_generator(self, views):
        view_weights = [layer(view) for layer, view in zip(self.attention_layers, views)]
        return torch.cat(view_weights, dim=-1)

# Squeeze-and-Excitation Attention
class MultiViewSEAttention(MultiViewAttention):
    def __init__(self, num_views, feature_dim, is_sharedspecific):
        super().__init__(num_views, feature_dim, is_sharedspecific)
        self.fc_layers = nn.Sequential(
            nn.Linear(num_views, num_views // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_views // 2, num_views),
            nn.Sigmoid()
        )

    def weight_generator(self, views):
        views_tensor = torch.stack(views, dim=1)  # (batch_size, num_views, feature_dim)
        avg_pooled = F.adaptive_avg_pool1d(views_tensor, 1).squeeze()  # (batch_size, num_views)
        return self.fc_layers(avg_pooled)  # (batch_size, num_views)

# Efficient Channel Attention
class MultiViewECAAttention(MultiViewAttention):
    def __init__(self, num_views, feature_dim, is_sharedspecific, kernel_size=3):
        super().__init__(num_views, feature_dim, is_sharedspecific)
        self.conv = nn.Conv1d(
            in_channels=num_views,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def weight_generator(self, views):
        views_tensor = torch.stack(views, dim=-1)  # (batch_size, feature_dim, num_views)
        conv_output = self.conv(views_tensor).squeeze(1)  # (batch_size, num_views)
        return torch.sigmoid(conv_output)  # (batch_size, num_views)

# Adaptive Multi-View Attention
class AdaptiveMultiViewAttention(MultiViewAttention):
    def __init__(self, num_views, feature_dim, is_sharedspecific):
        super().__init__(num_views, feature_dim, is_sharedspecific)
        self.cross_view_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4
        )
        self.query_linear = nn.Linear(feature_dim, feature_dim)
        self.key_linear = nn.Linear(feature_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim, feature_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(num_views, num_views // 2),
            nn.ReLU(),
            nn.Linear(num_views // 2, num_views),
            nn.Softmax(dim=-1)
        )

    def weight_generator(self, views):
        views_tensor = torch.stack(views)  # (num_views, batch_size, feature_dim)
        Q = self.query_linear(views_tensor)  # (num_views, batch_size, feature_dim)
        K = self.key_linear(views_tensor)  # (num_views, batch_size, feature_dim)
        V = self.value_linear(views_tensor)  # (num_views, batch_size, feature_dim)
        attn_output, _ = self.cross_view_attention(Q, K, V)  # (num_views, batch_size, feature_dim)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, num_views, feature_dim)
        attn_output_pooled = attn_output.mean(dim=2)  # (batch_size, num_views)
        return self.fc_layers(attn_output_pooled)  # (batch_size, num_views)


# MultiViewDSFusion
class MultiViewDSFusion(nn.Module):
    def __init__(self, num_views, feature_dim, num_classes):
        super().__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        # 使用 nn.Sequential 定义每个分类器的多个层
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, num_classes),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ) for _ in range(num_views)
        ])
        self.combiner = DirichletCombiner(classes=num_classes)

        self.fusion = nn.Linear(feature_dim * num_views, feature_dim) 

    def weight_generator(self, views):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, views):

        alphas = dict()
        for v_num in range(self.num_views):
            evidence = self.classifiers[v_num](views[v_num])
            alphas[v_num] = evidence + 1

        
        alpha_combined, u_a, u_list = self.combiner.DS_Combin(alphas)        
        evidence_a = alpha_combined - 1
        return alphas, alpha_combined, u_a, u_list
