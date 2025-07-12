import torch.nn as nn
from models.methods.reverse_gradient import DomainAdversarialNetwork
from models.encoder import FeatureEncoder2D, FeatureEncoder3D

from models.decoder import (
    ConcatDecoder,
    AttentionDecoder
)

class MultiViewFeatureFusion(nn.Module):  
    """  
    A multi-view feature fusion network combining RT, DT (2D), and RDT, ERT, ART (3D) features.  
    """  
    def __init__(self, backbone="custom", cnn_output_size=128, hidden_size=128,   
                 rnn_type='lstm', lstm_layers=1, bidirectional=True, fc_size=7, num_domains = 5,  
                 input_feature_shapes=None, fusion_mode='concatenate', method='add', is_sharedspecific=0,
                 bottleneck_dim=None, selected_features=None, is_domain_loss=0, num_classes=7):  
        super(MultiViewFeatureFusion, self).__init__()  

        self.fusion_mode = fusion_mode  # Fusion mode (concatenate/attention)  
        self.method = method  # (e.g., 'add', etc.)  
        self.bottleneck_dim = bottleneck_dim  # Bottleneck dimension for shared-specific methods  
        self.selected_features = selected_features if selected_features else ['RT', 'DT', 'RDT', 'ERT', 'ART']  
        self.num_views = len(self.selected_features)
        # Dynamically create encoders for the selected features  
        self.encoders = nn.ModuleDict()  
        for feature in self.selected_features:  
            if feature in ['RT', 'DT']:  # 2D Features  
                self.encoders[feature] = FeatureEncoder2D(  
                    fc_size=cnn_output_size, backbone=backbone, use_fc=True,  
                    input_feature_shape=input_feature_shapes[feature]  
                )  
            elif feature in ['RDT', 'ERT', 'ART']:  # 3D Features  
                self.encoders[feature] = FeatureEncoder3D(  
                    cnn_output_size=cnn_output_size, backbone=backbone, hidden_size=hidden_size,  
                    rnn_type=rnn_type, lstm_layers=lstm_layers, bidirectional=bidirectional,  
                    fc_size=cnn_output_size, feature_use_fc=True, input_feature_shape=input_feature_shapes[feature]  
                )  
            else:  
                raise ValueError(f"Unsupported feature type: {feature}. Choose from 'RT', 'DT', 'RDT', 'ERT', 'ART'.")  

        # Dimension of each feature representation (e.g., `cnn_output_size` after encoding)  
        feature_dim = cnn_output_size  

        # Initialize decoder based on fusion mode  
        if fusion_mode == 'concatenate':  
            # Use ConcatDecoder  
            self.decoder = ConcatDecoder(feature_dim, num_views=self.num_views, method=method)  
        elif fusion_mode == 'attention':  
            # Use AttentionDecoder with a specific attention strategy  
            self.decoder = AttentionDecoder(feature_dim, num_views=self.num_views, 
                                            is_sharedspecific=is_sharedspecific, method=method, num_classes=num_classes)   
        else:  
            raise ValueError(f"Invalid fusion mode: {fusion_mode}. Choose from 'concatenate' or 'attention'.")  

        # Fully connected layer for post-fusion representation  
        self.fc_fusion = nn.Sequential(  
            nn.Linear(feature_dim, fc_size),  
            nn.ReLU(inplace=True),  
            nn.Dropout()  
        )  
        self.classifier = nn.Sequential()
        # 定义一个域域判别器
        if is_domain_loss:
            self.domain_discriminator = DomainAdversarialNetwork(feature_dim=feature_dim, num_domains=num_domains,
                                                                 method=method, num_views=self.num_views)
    def forward(self, features):  
        """  
        Forward pass of the MultiViewFeatureFusion module.  

        Args:  
            features (dict): A dictionary where keys are feature names ('RT', 'DT', 'RDT', 'ERT', 'ART')   
                             and values are the corresponding raw features.  

        Returns:  
            torch.Tensor: The fused feature representation.  
        """  
        if not set(features.keys()).issubset(set(self.selected_features)):  
            raise ValueError(f"Input features ({features.keys()}) don't match selected features ({self.selected_features}).")  

        # Extract features using their corresponding encoders  
        encoded_features = []  
        for feature_name in self.selected_features:  
            encoded_feature = self.encoders[feature_name](features[feature_name])  
            encoded_features.append(encoded_feature)  

        # Fuse the extracted features using the decoder
        if self.method=='DScombine':
            alphas, alpha_combined, u_a, u_tensor = self.decoder(*encoded_features)  
            return encoded_features, alphas, alpha_combined, u_a, u_tensor
        else:  
            fused_features, weights = self.decoder(*encoded_features)  

            # Apply fully connected layer after fusion to obtain final representation  
            final_features = self.fc_fusion(fused_features)  
            ouputs = self.classifier(final_features)
            return ouputs, weights, fused_features