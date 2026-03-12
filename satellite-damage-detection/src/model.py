"""Model architectures for damage detection."""

import torch
import torch.nn as nn
import torchvision.models as models


class DamageClassifier(nn.Module):
    """ResNet-50 based damage classification model."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, input_channels: int = 3):
        """Initialize classifier.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary damage/no-damage)
            pretrained: Use ImageNet pretrained weights
            input_channels: Number of input channels (3 for RGB, 6+ for multispectral)
        """
        super().__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Adapt first layer if using multispectral input
        if input_channels != 3:
            # Average the pretrained weights across channels and expand
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize with averaged pretrained weights
            if pretrained:
                pretrained_weights = original_conv.weight
                new_conv.weight.data = pretrained_weights.mean(dim=1, keepdim=True).expand_as(new_conv.weight)
            
            resnet.conv1 = new_conv
        
        # Replace final layer for damage classification
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        self.model = resnet
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output logits (batch, num_classes)
        """
        return self.model(x)


class ChangeDetectionModel(nn.Module):
    """Model for temporal change detection."""
    
    def __init__(self, num_classes: int = 2):
        """Initialize change detection model.
        
        Args:
            num_classes: Number of classes (damage levels)
        """
        super().__init__()
        
        # Siamese-like architecture: process two timesteps
        self.encoder = models.resnet50(pretrained=True)
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # Change detection head
        self.change_head = nn.Sequential(
            nn.Linear(num_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> torch.Tensor:
        """Forward pass for temporal change detection.
        
        Args:
            x_t1: Image at time t1 (batch, channels, height, width)
            x_t2: Image at time t2 (batch, channels, height, width)
            
        Returns:
            Change classification (batch, num_classes)
        """
        # Encode both timesteps
        feat_t1 = self.encoder(x_t1)
        feat_t2 = self.encoder(x_t2)
        
        # Concatenate and classify change
        combined = torch.cat([feat_t1, feat_t2], dim=1)
        change_pred = self.change_head(combined)
        
        return change_pred


def create_model(model_type: str = 'damage_classifier', **kwargs) -> nn.Module:
    """Create a model.
    
    Args:
        model_type: Type of model ('damage_classifier' or 'change_detection')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'damage_classifier':
        return DamageClassifier(**kwargs)
    elif model_type == 'change_detection':
        return ChangeDetectionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
