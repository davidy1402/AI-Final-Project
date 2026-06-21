import torch
import torch.nn as nn
import torchvision.models as models

def get_flood_model(pretrained=True):
    """
    Constructs a ResNet-50 model adapted for binary classification (Flooded vs Not Flooded)
    using transfer learning.

    Freezes early layers of the network, leaving only the last two residual blocks
    (layer3 and layer4) and the final fully connected (FC) classification layer trainable.

    Args:
        pretrained (bool): Whether to load pre-trained ImageNet weights. Default is True.

    Returns:
        torch.nn.Module: The modified ResNet-50 model.
    """
    # Load ResNet-50 using modern API with fallback for older torchvision versions
    try:
        if pretrained:
            # Replaces pretrained=True in newer versions
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)
        
    # Freeze all layers by default
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the last 2 residual blocks
    # ResNet-50 structure: conv1/bn1 -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace the final fully connected layer for binary classification:
    # Class 0: Not Flooded, Class 1: Flooded
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Ensure the new fully connected layer has gradients enabled
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model
