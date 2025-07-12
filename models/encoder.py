import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FeatureEncoder2D(nn.Module):
    """
    适用于RT, DT 二维的特征
    Encoder for 2D feature inputs with optional FC or GAP.
    """
    def __init__(self, fc_size=128, backbone="custom", use_fc=True, input_feature_shape=(1, 64, 64)):
        """
        Args:
        - fc_size (int): Size of the fully connected output (if `use_fc=True`).
        - backbone (str): Choice of CNN backbone. Options: "custom", "alexnet", "resnet18", "resnet50", etc.
        - use_fc (bool): If True, use a fully connected layer. If False, use global average pooling.
        """
        super(FeatureEncoder2D, self).__init__()
        
        self.backbone = backbone.lower()
        self.use_fc = use_fc
        self.input_feature_shape = input_feature_shape  # 输入特征形状 (C, H, W)

        if self.backbone == "custom":
            # Default custom CNN
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(3, 6, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        else:
            if self.backbone != "lenet5":
                self.input_feature_shape[0] = 3 # 通道变成3
            # Use a pretrained CNN as the backbone
            self.feature_extractor = self._load_pretrained_backbone(self.backbone)
            
        # Dynamically determine the output size after the backbone
        self.feature_output_size = self._compute_feature_size(input_feature_shape)


        if self.use_fc:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_output_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout()
            )

    def _load_pretrained_backbone(self, backbone):
        """
        Load a CNN backbone without pretrained weights.
        """
        if backbone == "resnet18":
            # ResNet-18 without pretrained weights
            model = models.resnet18(weights=None)
            # Remove the fully connected layer (fc) and adaptive pooling
            feature_extractor = nn.Sequential(
                *list(model.children())[:-2]
            )
        elif backbone == "mobilenet":
            # MobileNetV2 without pretrained weights
            model = models.mobilenet_v3_small(weights=None)
            feature_extractor = model.features
        elif backbone == "lenet5":
            # Define a custom LeNet-5 model
            feature_extractor = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'lenet5', 'mobilenet', 'resnet18', 'custom'.")
        
        return feature_extractor
    
    def _compute_feature_size(self, input_feature_shape):
        """
        Compute the output size of the feature extractor for a given input shape.
        """
        C, H, W = input_feature_shape

        dummy_input = torch.randn(1, C, H, W)  # Create a dummy input tensor
        with torch.no_grad():
            output = self.feature_extractor(dummy_input)
        return output.view(1, -1).size(1)  # Flatten and get the size
    
    def forward(self, x):
        """
        Forward pass.
        - x: Input tensor of shape [batch_size, channels, height, width].
        """
        if self.backbone != "custom" and self.backbone != "lenet5":
            # Pretrained models expect 3 channels (RGB). Handle grayscale inputs by repeating the channel.
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

        x = self.feature_extractor(x)  # Feature extraction

        if self.use_fc:
            # Flatten spatial dimensions (H' * W') and apply FC
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            # Apply global average pooling (GAP) over spatial dimensions
            x = F.adaptive_avg_pool2d(x, (1, 1))  # Output shape (B, C', 1, 1)
            x = x.view(x.size(0), -1)  # Flatten to (B, C')

        return x


class FeatureEncoder3D(nn.Module):
    """
    适用于RDT,ART,ERT 三维的特征
    Process 3D input with 2D CNN + RNN/LSTM/GRU.
    """
    def __init__(self, cnn_output_size=128, backbone="custom", hidden_size=128, rnn_type='lstm', lstm_layers=1, bidirectional=True, fc_size=128, feature_use_fc=True, input_feature_shape=(1, 64, 64)):
        super(FeatureEncoder3D, self).__init__()
        input_feature_shape_2D = input_feature_shape[1:]
        # Use the existing FeatureEncoder2D class for CNN feature extraction
        self.feature_extractor = FeatureEncoder2D(fc_size=cnn_output_size, backbone=backbone, use_fc=feature_use_fc, input_feature_shape=input_feature_shape_2D)

        # Recurrent Layer
        rnn_input_size = cnn_output_size  # Flattened CNN output size
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=lstm_layers,
                               bidirectional=bidirectional, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=lstm_layers,
                              bidirectional=bidirectional, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=lstm_layers,
                              bidirectional=bidirectional, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        """
        Input: 
        - x: Tensor of shape [batch_size, seq_len, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = x.size()
        # Reshape to process each 2D frame independently through CNN
        x = x.view(batch_size * seq_len, channels, height, width)
        cnn_features = self.feature_extractor(x)  # Shape: [batch_size * seq_len, channels_out, h', w']
        
        # Flatten CNN features for RNN
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Shape: [batch_size * seq_len, cnn_output_size]
        cnn_output_size = cnn_features.size(1)  # Save this to define RNN input size
        
        # Reshape back for RNN input
        rnn_input = cnn_features.view(batch_size, seq_len, cnn_output_size)  # Shape: [batch_size, seq_len, cnn_output_size]
        
        # Pass through RNN
        rnn_output, _ = self.rnn(rnn_input)  # Shape: [batch_size, seq_len, hidden_size * num_directions]
        
        # Use the last time step's output
        final_output = rnn_output[:, -1, :]  # Shape: [batch_size, hidden_size * num_directions]
        
        # Fully connected layer
        output = self.fc(final_output)  # Shape: [batch_size, fc_size]
        return output
