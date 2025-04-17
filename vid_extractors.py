import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import models

class EfficientNetV2FeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Load a pre-trained EfficientNetV2-S model
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        
        # Remove the classification head
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # Exclude the classifier
        self.feature_dim = model.classifier[1].in_features  # Input features to the classifier

        # Projection layer to desired output dimension
        self.projector = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):  # x shape: [B, T, 3, 160, 160]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten batch and time dimensions

        feats = self.backbone(x)  # Extract features
        feats = feats.view(B * T, self.feature_dim)  # Flatten spatial dimensions

        projected = self.projector(feats)  # Project to desired dimension
        output = projected.view(B, T, -1)  # Reshape back to [B, T, output_dim]
        return output


class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)

        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = model.classifier[0].in_features

        self.projector = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)
        feats = feats.view(B * T, self.feature_dim)

        projected = self.projector(feats)
        output = projected.view(B, T, -1)
        return output


class ViTFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Load a pre-trained ViT model
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.feature_dim = self.vit.hidden_dim  # Typically 768 for vit_b_16

        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Optional: add a projection layer if output_dim differs from feature_dim
        if output_dim != self.feature_dim:
            self.projector = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, x):  # x shape: [B, T, 3, 160, 160]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten batch and time dimensions

        # Process input through ViT
        feats = self.vit(x)  # Output shape: [B*T, feature_dim]

        # Project features if necessary
        projected = self.projector(feats)  # Shape: [B*T, output_dim]

        # Reshape back to [B, T, output_dim]
        output = projected.view(B, T, -1)
        return output

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Load a pretrained ResNet and remove the classifier
        resnet = models.resnet18(pretrained=True)
        self.feature_dim = resnet.fc.in_features  

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output shape: [B*T, 2048, 1, 1]

        self.projector = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):  # x shape: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]

        feats = self.backbone(x)  # [B*T, 2048, 1, 1]
        feats = feats.view(B * T, self.feature_dim)  # flatten: [B*T, 2048]

        projected = self.projector(feats)  # [B*T, 768]
        output = projected.view(B, T, -1)  # reshape back to [B, T, 768]
        return output