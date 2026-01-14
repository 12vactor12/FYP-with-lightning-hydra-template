# 兰花分类项目技术文档

## 1. 项目概述

### 1.1 项目简介
本项目是一个基于PyTorch Lightning和Hydra的兰花分类系统，支持多种深度学习模型架构，包括ResNet50、InceptionV3、ViT等，用于对兰花图像进行分类识别。

### 1.2 主要功能
- 支持多种模型架构的兰花分类
- 完整的训练、验证和测试流程
- 自动模型评估和指标记录
- TSNE特征可视化
- Grad-CAM可视化解释
- 灵活的配置管理
- 支持预训练权重
- 支持模型编译（PyTorch 2.0+）

## 2. 项目架构与设计

### 2.1 架构图
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   配置层 (Hydra) │────▶│   训练层        │────▶│   模型层        │
│   configs/      │     │   src/train.py  │     │   src/models/   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          ▲                        ▲                        │
          │                        │                        │
          │                        │                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   数据层        │────▶│   评估层        │     │   可视化层      │
│   src/data/     │     │   src/eval.py   │     │   src/visualization/ │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 2.2 设计原则
- **模块化设计**：将数据处理、模型定义、训练逻辑分离
- **配置驱动**：使用Hydra管理所有配置，支持动态配置
- **可扩展性**：易于添加新模型架构和功能
- **可重现性**：支持固定随机种子，确保实验可重现
- **可视化支持**：内置TSNE和Grad-CAM可视化

## 3. 模块与文件功能

### 3.1 配置层 (configs/)

| 文件/目录 | 功能描述 |
|---------|---------|
| configs/callbacks/ | 训练回调配置（模型检查点、早停、TSNE可视化等） |
| configs/data/ | 数据模块配置 |
| configs/debug/ | 调试配置 |
| configs/experiment/ | 实验配置 |
| configs/logger/ | 日志记录器配置 |
| configs/model/ | 模型配置 |
| configs/paths/ | 路径配置 |
| configs/trainer/ | 训练器配置 |
| configs/train.yaml | 主训练配置 |
| configs/eval.yaml | 评估配置 |

### 3.2 核心代码层 (src/)

#### 3.2.1 训练与评估

| 文件 | 功能描述 |
|------|---------|
| src/train.py | 训练入口，负责模型训练和测试 |
| src/eval.py | 评估入口，负责模型评估 |

#### 3.2.2 数据模块

| 文件 | 功能描述 |
|------|---------|
| src/data/my_dataset_datamodule.py | 自定义数据集的数据模块，负责数据加载和预处理 |
| src/data/components/ | 数据组件 |

#### 3.2.3 模型模块

| 文件 | 功能描述 |
|------|---------|
| src/models/orchid_module.py | 兰花分类的LightningModule，实现训练、验证和测试逻辑 |
| src/models/components/orchid_models.py | 多种模型架构的实现，支持ResNet50、ViT等 |

#### 3.2.4 工具模块

| 文件 | 功能描述 |
|------|---------|
| src/utils/instantiators.py | 实例化工具，用于创建回调和日志记录器 |
| src/utils/logging_utils.py | 日志工具 |
| src/utils/pylogger.py | 日志记录器实现 |
| src/utils/rich_utils.py | Rich库相关工具 |
| src/utils/tsne_callback.py | TSNE可视化回调 |
| src/utils/utils.py | 通用工具函数 |

#### 3.2.5 可视化模块

| 文件 | 功能描述 |
|------|---------|
| src/visualization/generate_gradcam.py | Grad-CAM可视化生成脚本 |
| src/visualization/gradcam_visualization.py | Grad-CAM可视化实现 |
| src/visualization/multi_model_visualization.py | 多模型可视化比较 |
| src/visualization/tsne_visualization.py | TSNE可视化脚本 |

## 4. 主要类与核心函数

### 4.1 核心类

#### 4.1.1 OrchidLitModule
**位置**: `src/models/orchid_module.py`

**功能**: 兰花分类的LightningModule，实现完整的训练、验证和测试流程。

**主要方法**:
- `__init__`: 初始化模型、优化器、调度器和指标
- `forward`: 模型前向传播
- `model_step`: 单个模型步骤，计算损失和预测
- `training_step`: 训练步骤
- `validation_step`: 验证步骤
- `test_step`: 测试步骤
- `configure_optimizers`: 配置优化器和调度器

#### 4.1.2 OrchidModel
**位置**: `src/models/components/orchid_models.py`

**功能**: 灵活的模型类，支持多种模型架构。

**主要方法**:
- `__init__`: 初始化模型
- `_get_backbone`: 根据模型名称获取backbone
- `_get_classifier`: 获取分类器
- `forward`: 模型前向传播
- `get_features`: 提取特征（用于可视化）

### 4.2 核心函数

#### 4.2.1 train
**位置**: `src/train.py`

**功能**: 训练模型，并可选地在测试集上进行评估。

**参数**:
- `cfg`: Hydra配置字典

**返回值**:
- 包含训练和测试指标的字典
- 包含所有实例化对象的字典

#### 4.2.2 evaluate
**位置**: `src/eval.py`

**功能**: 使用指定的检查点对数据模块的测试集进行评估。

**参数**:
- `cfg`: Hydra配置字典

**返回值**:
- 包含评估指标的字典
- 包含所有实例化对象的字典

#### 4.2.3 TSNEVisualizationCallback
**位置**: `src/utils/tsne_callback.py`

**功能**: 训练过程中生成TSNE可视化。

**主要方法**:
- `on_validation_epoch_end`: 验证epoch结束时生成可视化
- `generate_tsne`: 生成TSNE可视化
- `visualize_tsne`: 可视化TSNE结果

## 5. 模块间调用关系

### 5.1 训练流程

```
src/train.py:main()
    ├── src/train.py:train()
    │   ├── hydra.utils.instantiate(cfg.data) → DataModule
    │   ├── hydra.utils.instantiate(cfg.model) → Model
    │   ├── instantiate_callbacks(cfg.callbacks) → Callbacks
    │   ├── instantiate_loggers(cfg.logger) → Loggers
    │   ├── hydra.utils.instantiate(cfg.trainer) → Trainer
    │   ├── trainer.fit()
    │   └── trainer.test()
    └── get_metric_value()
```

### 5.2 模型训练流程

```
trainer.fit()
    ├── on_train_start()
    ├── training_step()
    │   ├── model_step()
    │   ├── update metrics
    │   └── log metrics
    ├── on_train_epoch_end()
    ├── validation_step()
    │   ├── model_step()
    │   ├── update metrics
    │   └── log metrics
    └── on_validation_epoch_end()
        └── update best metrics
```

### 5.3 可视化流程

```
# TSNE可视化
src/utils/tsne_callback.py:on_validation_epoch_end()
    ├── generate_tsne()
    │   ├── extract_features()
    │   └── visualize_tsne()

# Grad-CAM可视化
src/visualization/generate_gradcam.py:main()
    ├── load model from checkpoint
    ├── for each image:
    │   ├── preprocess_image()
    │   ├── generate_gradcam()
    │   └── visualize_and_save_gradcam()
```

## 6. 关键业务流程

### 6.1 模型训练流程

1. **配置加载**: 使用Hydra加载配置文件
2. **数据准备**: 实例化数据模块，准备训练、验证和测试数据
3. **模型初始化**: 实例化模型，支持预训练权重
4. **优化器配置**: 配置优化器和学习率调度器
5. **训练执行**: 执行模型训练，记录训练指标
6. **验证评估**: 每个epoch结束后进行验证，更新最佳模型
7. **测试评估**: 训练完成后使用最佳模型进行测试
8. **结果保存**: 保存模型检查点和训练日志

### 6.2 模型评估流程

1. **配置加载**: 使用Hydra加载评估配置
2. **模型加载**: 从检查点加载预训练模型
3. **数据准备**: 实例化数据模块，准备测试数据
4. **评估执行**: 执行模型评估，计算评估指标
5. **结果记录**: 记录评估结果

### 6.3 可视化生成流程

1. **TSNE可视化**: 训练过程中自动生成，每N个epoch执行一次
2. **Grad-CAM可视化**: 训练完成后手动触发，可选择单张或多张图像

## 7. 外部接口与第三方库

### 7.1 核心依赖

| 库名称 | 版本 | 用途 |
|--------|------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 计算机视觉工具库 |
| lightning | >=2.0.0 | PyTorch Lightning框架 |
| hydra-core | ==1.3.2 | 配置管理框架 |
| timm | - | 预训练模型库 |
| torchmetrics | >=0.11.4 | 模型评估指标 |
| pytorch-grad-cam | >=1.4.0 | Grad-CAM可视化 |
| scikit-learn | - | 机器学习工具库 |
| matplotlib | - | 可视化库 |
| pillow | - | 图像处理库 |

### 7.2 配置管理

使用Hydra进行配置管理，支持：
- 分层配置
- 命令行参数覆盖
- 配置组合
- 多运行实验

### 7.3 日志记录

支持多种日志记录器：
- TensorBoard
- Wandb
- Comet
- Neptune
- MLflow

## 8. 模型架构

### 8.1 支持的模型

| 模型名称 | 架构类型 | 预训练支持 |
|----------|----------|------------|
| resnet50 | CNN | ✓ |
| inception_v3 | CNN | ✓ |
| vit_base_16 | Transformer | ✓ |
| bilinear_cnn | CNN | ✓ |
| mae_vit | Transformer | ✓ |
| dino_vit | Transformer | ✓ |
| ibot_vit | Transformer | ✓ |

### 8.2 模型结构

所有模型遵循统一的结构：
- **Backbone**: 特征提取网络
- **Classifier**: 分类头
- **特征提取**: 支持获取中间特征（用于可视化）

## 9. 数据处理

### 9.1 数据格式

```
data/
└── my_dataset/
    ├── 类别1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── 类别2/
        ├── image3.jpg
        └── image4.jpg
```

### 9.2 数据预处理

- 图像大小调整: 224x224
- 归一化: 均值 [0.485, 0.456, 0.406], 标准差 [0.229, 0.224, 0.225]
- 数据增强: 随机水平翻转、随机裁剪等

## 10. 可视化功能

### 10.1 TSNE可视化

- **触发时机**: 每N个验证epoch结束后
- **实现**: 使用scikit-learn的TSNE算法
- **结果**: 保存到 `visualizations/tsne/` 目录

### 10.2 Grad-CAM可视化

- **触发方式**: 手动执行脚本
- **实现**: 使用pytorch-grad-cam库
- **结果**: 保存到 `visualizations/gradcam/` 目录

## 11. 部署与使用

### 11.1 环境搭建

```bash
# 使用conda创建环境
conda env create -f environment.yaml
conda activate FYPWithLightning

# 或使用pip安装依赖
pip install -r requirements.txt
```

### 11.2 训练模型

```bash
# 使用默认配置训练
python src/train.py

# 使用指定实验配置训练
python src/train.py experiment=resnet50

# 使用GPU训练
python src/train.py trainer=gpu
```

### 11.3 评估模型

```bash
python src/eval.py --ckpt_path <checkpoint_path>
```

### 11.4 生成可视化

```bash
# 生成Grad-CAM可视化
python src/visualization/generate_gradcam.py --ckpt_path <checkpoint_path> --image_path <image_path>
```

## 12. 代码规范与开发指南

### 12.1 代码风格

- 遵循Google Python风格指南
- 使用pre-commit进行代码检查
- 函数和类使用文档字符串
- 变量和函数名使用snake_case
- 类名使用CamelCase

### 12.2 文档规范

- 模块级文档字符串: 描述模块功能和主要类
- 类文档字符串: 描述类的功能和使用方法
- 函数文档字符串: 描述函数的参数、返回值和作用
- 复杂逻辑添加行内注释

### 12.3 测试

```bash
# 运行快速测试
make test

# 运行所有测试
make test-full
```

## 13. 扩展与维护

### 13.1 添加新模型

1. 在 `src/models/components/orchid_models.py` 中添加新模型支持
2. 在 `_get_backbone` 方法中添加模型创建逻辑
3. 调整 `forward` 和 `get_features` 方法以支持新模型
4. 在 `configs/model/` 中添加对应的配置文件

### 13.2 添加新数据集

1. 继承 `LightningDataModule` 创建新的数据模块
2. 实现 `setup`, `train_dataloader`, `val_dataloader`, `test_dataloader` 方法
3. 在 `configs/data/` 中添加对应的配置文件

### 13.3 添加新可视化

1. 创建新的可视化脚本或回调
2. 实现可视化逻辑
3. 集成到训练流程或提供独立脚本

## 14. 监控与日志

### 14.1 训练监控

- 使用TensorBoard监控训练过程
- 支持多种指标监控：损失、准确率、精确率、召回率、F1分数
- 记录最佳验证指标

### 14.2 日志记录

- 训练日志保存在 `logs/train/runs/` 目录
- 每个运行创建独立的日志目录
- 包含配置文件、检查点和训练日志

## 15. 总结与展望

### 15.1 项目亮点

- 模块化设计，易于扩展
- 配置驱动，灵活可调
- 支持多种模型架构
- 完整的训练和评估流程
- 强大的可视化支持
- 良好的代码规范和文档

### 15.2 未来改进方向

- 支持更多模型架构
- 添加更多数据增强策略
- 实现模型蒸馏
- 支持多模态输入
- 部署为Web服务
- 添加更多可视化方法

---

**文档版本**: v1.0  
**创建日期**: 2026-01-07  
**维护人**: -