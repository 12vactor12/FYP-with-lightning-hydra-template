import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional

class OrchidModel(nn.Module):
    """A flexible model class that supports multiple architectures for orchid classification."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize an OrchidModel with the specified architecture.
        
        :param model_name: The name of the model architecture to use.
        :param num_classes: The number of output classes for classification.
        :param pretrained: Whether to use pretrained weights.
        :param freeze_backbone: Whether to freeze the backbone layers.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Initialize the model based on the specified architecture
        self.backbone, self.embed_dim = self._get_backbone()
        self.classifier = self._get_classifier()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _get_backbone(self):
        """Get the backbone model based on the specified architecture."""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=self.pretrained)
            embed_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, embed_dim
        
        elif self.model_name == "inception_v3":
            model = models.inception_v3(pretrained=self.pretrained, aux_logits=True)
            embed_dim = model.fc.in_features
            model.fc = nn.Identity()
            model.AuxLogits.fc = nn.Identity()
            return model, embed_dim
        
        elif self.model_name == "vit_base_16":
            model = timm.create_model("vit_base_patch16_224", pretrained=self.pretrained)
            embed_dim = model.head.in_features
            model.head = nn.Identity()
            return model, embed_dim
        
        elif self.model_name == "bilinear_cnn":
            # Bilinear CNN implementation
            resnet = models.resnet50(pretrained=self.pretrained)
            # Remove the last two layers (avgpool and fc)
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            embed_dim = 2048 * 2048  # Bilinear pooling output
            return backbone, embed_dim
        
        elif self.model_name == "mae_vit":
            model = timm.create_model("vit_base_patch16_224.mae", pretrained=self.pretrained, num_classes=0)
            embed_dim = model.num_features
            return model, embed_dim
        
        elif self.model_name == "dino_vit":
            model = timm.create_model("vit_base_patch16_224.dino", pretrained=self.pretrained, num_classes=0)
            embed_dim = model.num_features
            return model, embed_dim
        
        # 如果用户选择 iBot-ViT 作为骨干网络
        elif self.model_name == "ibot_vit":
            # 通过 timm 创建 ViT-Base/16 224px 的 iBot 预训练版本
            model = timm.create_model("vit_base_patch16_224.ibot", pretrained=self.pretrained, num_classes=0)
            # 取出原分类头的输入维度，作为后续自定义分类层的 embed_dim
            embed_dim = model.num_features
            # 返回特征提取网络与对应的特征维度
            return model, embed_dim
        
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
    
    def _get_classifier(self):
        """Get the classifier based on the model architecture."""
        if self.model_name == "bilinear_cnn":
            # For Bilinear CNN, we need a special classifier
            return nn.Linear(self.embed_dim, self.num_classes)
        else:
            # Standard classifier for other models
            return nn.Linear(self.embed_dim, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        if self.model_name == "bilinear_cnn":
            # Bilinear CNN forward pass with bilinear pooling
            features = self.backbone(x)
            batch_size, channels, height, width = features.size()
            
            # Reshape features for bilinear pooling
            features = features.view(batch_size, channels, height * width)
            
            # Bilinear pooling: (batch, channels, hw) x (batch, channels, hw) -> (batch, channels^2)
            bilinear_features = torch.bmm(features, features.transpose(1, 2))
            bilinear_features = bilinear_features.view(batch_size, -1)
            
            # Normalization
            bilinear_features = torch.sign(bilinear_features) * torch.sqrt(torch.abs(bilinear_features) + 1e-5)
            bilinear_features = nn.functional.normalize(bilinear_features, dim=1)
            
            # Classification
            x = self.classifier(bilinear_features)
        elif self.model_name == "inception_v3":
            # InceptionV3 forward pass handles aux logits
            features = self.backbone(x)
            if isinstance(features, tuple):
                # During training, InceptionV3 returns (main_output, aux_output)
                features = features[0]
            x = self.classifier(features)
        else:
            # Standard forward pass for other models
            features = self.backbone(x)
            x = self.classifier(features)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model without classification.
        
        :param x: The input tensor.
        :return: A tensor of features.
        """
        if self.model_name == "bilinear_cnn":
            features = self.backbone(x)
            batch_size, channels, height, width = features.size()
            features = features.view(batch_size, channels, height * width)
            bilinear_features = torch.bmm(features, features.transpose(1, 2))
            bilinear_features = bilinear_features.view(batch_size, -1)
            bilinear_features = torch.sign(bilinear_features) * torch.sqrt(torch.abs(bilinear_features) + 1e-5)
            features = nn.functional.normalize(bilinear_features, dim=1)
        elif self.model_name == "inception_v3":
            features = self.backbone(x)
            if isinstance(features, tuple):
                # During training, InceptionV3 returns (main_output, aux_output)
                features = features[0]
        else:
            features = self.backbone(x)
        
        return features

if __name__ == "__main__":
    # Test the model with different architectures
    for model_name in ["resnet50", "vit_base_16", "mae_vit", "dino_vit"]:
        print(f"Testing {model_name}...")
        model = OrchidModel(model_name=model_name, num_classes=10, pretrained=False)
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)
        features = model.get_features(input_tensor)
        print(f"  Output shape: {output.shape}")
        print(f"  Features shape: {features.shape}")
        print()
