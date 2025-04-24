import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models
# from pytorch_i3d.pytorch_i3d import InceptionI3d
from torch import Tensor
import torch
import timm
from torch import nn
import torch.utils.checkpoint
import contextlib
import torchvision
from einops import rearrange
import math
from transformers import MT5ForConditionalGeneration, T5Tokenizer 
import warnings
from config import mt5_path
class i3d(nn.Module):
    def __init__(self, output_dim=768, freeze_vision_encoder=False):
        super().__init__()

        self.video_proj = nn.Linear(1024, 768)
        self.i3d_encoder = InceptionI3d(num_classes=400, in_channels=3)
        i3d_pretrained_path ='pytorch_i3d/models/rgb_imagenet.pt'
        self.i3d_encoder.load_state_dict(torch.load(i3d_pretrained_path), strict=False)
        self.i3d_encoder.avg_pool = nn.Identity()
        self.i3d_encoder.logits = nn.Identity()
        if freeze_vision_encoder:
            for param in self.i3d_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        video_i3d = x.permute(0, 2, 1, 3, 4) # From (B, T, 3, 128, 128) -> (B, 3, T, 128, 128)
        features = self.i3d_encoder(video_i3d) # The output shape might be (B, feature_dim, T_i3d, H_i3d, W_i3d)b, 1024,32, 7, 7 
        features = features.mean(dim=[-2, -1])  # Now (B, feature_dim, T_i3d)
        features = features.transpose(1, 2)
        T_new = features.shape[1]  # New temporal dimension from I3D
        # Project features to 768 dimensions.
        video_embeds = self.video_proj(features)
        return video_embeds

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
    def __init__(self, model_name='vit_small_patch16_224', output_dim=384, freeze_vision_encoder=False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.feature_dim = self.vit.num_features

        self.vit.reset_classifier(0)  # Remove the classification head

        if freeze_vision_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False

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
class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_tiny', output_dim=768, freeze_vision_encoder=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features  # e.g., 768 for tiny, 1024 for base, 1536 for large

        if freeze_vision_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if output_dim != self.feature_dim:
            self.projector = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, x):  # x shape: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)  # [B*T, feature_dim]
        projected = self.projector(feats)
        return projected.view(B, T, -1)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768, freeze_vision_encoder = False):
        super().__init__()
        # Load a pretrained ResNet and remove the classifier
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_dim = resnet.fc.in_features  

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output shape: [B*T, 512, 1, 1] for ResNet18

        if freeze_vision_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projector = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):  # x shape: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]

        feats = self.backbone(x)  # [B*T, 2048, 1, 1]
        feats = feats.view(B * T, self.feature_dim)  # flatten: [B*T, 2048]

        projected = self.projector(feats)  # [B*T, 768]
        output = projected.view(B, T, -1)  # reshape back to [B, T, 768]
        return output