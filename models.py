# models.py
import torch
import torch.nn as nn
from tiny_model import SuperTinyMammoNet  # Use YOUR model, not ResNet50

def get_model(weights_path, model_type='supertiny'):
    """
    Load your trained model with weights
    
    Args:
        weights_path: Path to .pth file
        model_type: 'supertiny' (your model) or 'resnet50'
    
    Returns:
        Loaded PyTorch model
    """
    if model_type == 'supertiny':
        # Use YOUR model
        model = SuperTinyMammoNet(num_classes=2)
    else:
        # Optional: ResNet50 (if you ever want to try it)
        from torchvision import models
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def get_target_layer(model, model_type='supertiny'):
    """
    Get the appropriate target layer for Grad-CAM
    """
    if model_type == 'supertiny':
        # For SuperTinyMammoNet, use last conv layer
        return model.features[-4]  # Last Conv2d layer
    else:
        # For ResNet50
        return model.layer4[-1]