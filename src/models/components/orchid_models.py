"""兰花分类模型组件

该模块定义了支持多种模型架构的兰花分类模型，包括：
- ResNet50
- ViT Base 16
- MAE ViT
- DINO ViT
- iBot ViT

主要功能：
- 根据模型名称动态创建不同架构的模型
- 支持预训练权重
- 支持冻结 backbone
- 支持特征提取
- 支持多种分类头配置
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional


class OrchidModel(nn.Module):
    """灵活的兰花分类模型类，支持多种模型架构。
    
    该类可以根据指定的模型名称动态创建不同架构的模型，
    支持预训练权重、backbone冻结等功能，适合用于兰花分类任务。
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_16",
        num_classes: int = 19,
        pretrained: bool = True,
        freeze_backbone: str = "none",  # "none": 全量微调, "full": 完全冻结, "partial": 部分微调
        classifier:str = "linear",
        supcon:bool = False,
        proj_dim: int = 128
    ) -> None:
        """初始化OrchidModel。
        
        Args:
            model_name: 模型架构名称，支持 vit_base_16、dino_vit、resnet50
            num_classes: 分类的类别数量，默认为19
            pretrained: 是否使用预训练权重，默认为True
            freeze_backbone: 冻结模式，支持 "none"(全量微调)、"full"(完全冻结)、"partial"(部分微调)
            classifier: 分类头类型，默认为"linear"
            supercon: 是否使用对比学习，默认为False
            proj_dim: 对比学习投影维度，默认为128
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.classifier_type = classifier
        self.supercon = supcon
        self.proj_dim = proj_dim
        self.proj_head = None
        
        # 根据指定架构初始化模型
        self.backbone, self.embed_dim = self._get_backbone()

        if self.supercon:
            self.projection_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.proj_dim)
            )
        else:
            self.projection_head = None

        self.classifier = self._get_classifier()
        
        # 根据冻结模式设置backbone参数
        self._setup_backbone_freeze()
    
    def _setup_backbone_freeze(self):
        """根据冻结模式设置backbone的参数可训练性。
        
        - "none": 全量微调，所有参数均可训练
        - "full": 完全冻结，所有参数不可训练
        - "partial": 部分微调，根据模型类型解冻特定层
            - ResNet50: 解冻layer4
            - ViT Base 16/ViT Base 16 DINO: 解冻blocks10和11
        """
        if self.freeze_backbone == "none":
            # 全量微调，所有参数均可训练
            return
        elif self.freeze_backbone == "full":
            # 完全冻结，所有参数不可训练
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.freeze_backbone == "partial":
            # 部分微调，先冻结所有参数，再解冻特定层
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if self.model_name == "resnet50":
                # ResNet50: 解冻layer4
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif self.model_name in ["vit_base_16", "vit_base_16_dino"]:
                # ViT Base 16/ViT Base 16 DINO: 解冻blocks10和11
                for i in [10, 11]:
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = True
        else:
            raise ValueError(f"不支持的冻结模式: {self.freeze_backbone}")
    
    def _get_backbone(self):
        """根据指定的架构获取backbone模型。
        
        Returns:
            一个元组，包含：
            - backbone: 特征提取网络
            - embed_dim: 特征嵌入维度
        
        Raises:
            ValueError: 如果指定了不支持的模型名称
        """
        if self.model_name == "vit_base_16":
            # ViT Base 16模型
            model = timm.create_model("vit_base_patch16_224", pretrained=self.pretrained)
            embed_dim = model.head.in_features
            model.head = nn.Identity()  # 移除原始分类头
            return model, embed_dim
        
        elif self.model_name == "vit_base_16_dino":
            # DINO预训练的ViT模型
            model = timm.create_model("vit_base_patch16_224.dino", pretrained=self.pretrained, num_classes=0)
            embed_dim = model.num_features
            return model, embed_dim
        
        elif self.model_name == "resnet50":
            model = timm.create_model("resnet50", pretrained=self.pretrained)
            embed_dim = model.fc.in_features
            model.fc = nn.Identity()  # 移除原始分类头
            return model, embed_dim
        else:
            raise ValueError(f"不支持的模型名称: {self.model_name}")
    
    def _get_classifier(self):
        """根据模型架构获取分类器。
        
        Returns:
            分类器网络
        """
        if self.classifier_type == "linear":
            # 使用标准的线性分类器
            return nn.Linear(self.embed_dim, self.num_classes)
        elif self.classifier_type == "identity":
            # 返回恒等映射，用于特征提取
            return nn.Identity()
        else:
            raise ValueError(f"不支持的分类头类型: {self.classifier_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行模型的前向传播。
        
        Args:
            x: 输入图像张量，形状为 [batch_size, channels, height, width]

        Returns:
            模型输出的对数概率（logits），形状为 [batch_size, num_classes]
        """
        # 其他模型的标准前向传播
        features = self.backbone(x)
        x = self.classifier(features)
        return x
    
    def forward_supcon(self, x: torch.Tensor) -> torch.Tensor:
        assert self.projection_head is not None, "SupCon not enabled"
        features = self.backbone(x)
        z = self.projection_head(features)
        z = nn.functional.normalize(z, dim=1)
        return z

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """从模型中提取特征，不进行分类。
        
        Args:
            x: 输入图像张量，形状为 [batch_size, channels, height, width]

        Returns:
            提取的特征张量
        """
        # 其他模型的标准特征提取
        features = self.backbone(x)
        
        return features


if __name__ == "__main__":
    # 测试不同模型架构和冻结模式
    for model_name in ["resnet50", "vit_base_16", "vit_base_16_dino"]:
        for freeze_mode in ["none", "full", "partial"]:
            print(f"测试 {model_name}, 冻结模式: {freeze_mode}...")
            model = OrchidModel(model_name=model_name, num_classes=19, pretrained=False, freeze_backbone=freeze_mode)
            input_tensor = torch.randn(2, 3, 224, 224)
            output = model(input_tensor)
            features = model.get_features(input_tensor)
            print(f"  输出形状: {output.shape}")
            print(f"  特征形状: {features.shape}")
            
            # 统计可训练参数数量
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
            print()
