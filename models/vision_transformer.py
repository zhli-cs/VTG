import torch
import torch.nn as nn
import timm

class ViTCustom(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=4, embed_dim=512, depth=12, num_heads=16):
        super(ViTCustom, self).__init__()

        self.vit = timm.models.vision_transformer.VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=4.0, num_classes=num_classes
        )
    
    def forward(self, x):
        features = self.vit.forward_features(x)  
        cls_feature = features[:, 0, :] 
        logits = self.vit.head(cls_feature)  
        return cls_feature, logits

