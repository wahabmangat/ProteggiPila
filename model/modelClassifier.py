import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_v2_b

class BoxClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_swin: bool = True):
        super(BoxClassifier, self).__init__()
        
        # Load Swin Transformer backbone
        self.swin = swin_v2_b(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = self.swin.head.in_features  # Get output feature dimension
        self.swin.head = nn.Identity()  # Remove default classification head

        # Optionally freeze Swin Transformer
        if freeze_swin:
            for param in self.swin.parameters():
                param.requires_grad = False
        
        # Regularization and normalization
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.batch_norm = nn.BatchNorm1d(feature_dim)  # Add batch normalization
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        # Classifier with multiple layers
        self.fc1 = nn.Linear(feature_dim, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Final output layer for classification
        
    def forward(self, x):
        # Extract features using Swin Transformer
        features = self.swin(x)
        
        # # Apply Global Average Pooling (GAP)
        # x = self.global_avg_pool(features)
        # x = x.view(x.size(0), -1)  # Flatten the output to 2D tensor (batch_size, feature_dim)

        # # Apply batch normalization and dropout
        # x = self.batch_norm(x)
        x = self.dropout(features)
        
        # Classifier
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first fully connected layer
        class_out = self.fc2(x)  # Output the final classification result
        
        return class_out
