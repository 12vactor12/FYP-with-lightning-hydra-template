# src/data/my_dataset_datamodule.py
"""自定义数据集数据模块，用于封装数据加载逻辑。

该模块负责：
- 加载本地图像文件夹数据集
- 按指定比例划分训练/验证/测试集
- 应用不同的数据增强策略
- 提供对应的数据加载器
"""

from typing import Optional, List
import math
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from lightning import LightningDataModule

class TwoCropDataset(Dataset):
    """SupCon 使用：返回同一图像的两个随机增强视图"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return (view1, view2), label

class MyDatasetDataModule(LightningDataModule):
    """自定义数据集 LightningDataModule，用于封装数据加载逻辑。

    本模块负责：
    - 加载本地图像文件夹数据集
    - 按指定比例划分训练/验证/测试集
    - 应用不同的数据增强策略
    - 提供对应的数据加载器
    """

    def __init__(
        self,
        data_dir: str = "data/my_dataset/",
        batch_size: int = 32,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        pin_memory: bool = True,
        image_size: int = 224,
        seed: int = 42,
        train_subset_ratio: float = 1.0,
        supcon: bool = False
    ):
        """初始化 MyDatasetDataModule。

        Args:
            data_dir: 图像数据根目录，需符合 ImageFolder 结构。
            batch_size: 每个批次的样本数量。
            train_val_test_split: 训练/验证/测试集划分比例。
            num_workers: DataLoader 使用的子进程数量。
            pin_memory: 是否将张量固定在 GPU 内存中以加速传输。
            image_size: 输入网络的图像尺寸（正方形）。
            seed: 随机种子，用于固定数据集划分结果。
            train_subset_ratio: 训练集子集比例，范围(0, 1]，用于控制实际使用的训练数据量。
        """
        super().__init__()
        self.save_hyperparameters()

        # 训练阶段的数据增强变换
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        # 验证/测试阶段的标准化变换（无数据增强）
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 数据集对象占位符
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Lightning 钩子：下载或预处理数据（仅主进程执行）。

        本示例假设数据已存在，无需额外下载。
        """
        # 数据已存在于本地，无需下载或预处理

    def setup(self, stage: Optional[str] = None):
        """设置数据集，执行数据划分和子集抽样。

        Args:
            stage: 当前训练阶段，可选值为 'fit'、'validate'、'test' 或 'predict'。
                用于确定需要加载哪些数据集。
        """
        # 如果数据集已加载，直接返回
        if self.train_dataset is not None:
            return

        # Step 1: 创建索引数据集，用于获取样本标签和索引映射
        index_dataset = ImageFolder(
            root=self.hparams.data_dir,
            transform=None  # 暂不应用变换，仅用于获取索引和标签
        )
        all_targets = index_dataset.targets

        # Step 2: 构建类别到样本索引的映射
        class_to_sample_indices = defaultdict(list)
        for sample_idx, label in enumerate(all_targets):
            class_to_sample_indices[label].append(sample_idx)

        # 初始化各数据集索引列表
        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        # 设置随机种子生成器，确保划分结果可复现
        generator = torch.Generator().manual_seed(self.hparams.seed)

        # 解析划分比例
        train_ratio, val_ratio, _ = self.hparams.train_val_test_split

        # Step 3: 按类别均匀划分数据集
        for label, sample_indices in class_to_sample_indices.items():
            # 打乱当前类别的样本索引
            shuffled_indices = [sample_indices[i] for i in torch.randperm(
                len(sample_indices), generator=generator).tolist()]

            # 计算当前类别的样本总数
            num_samples = len(shuffled_indices)

            # 按比例计算各类别在不同数据集中的样本数量
            # 确保每个类别至少有一个样本分配到每个数据集
            num_train = max(1, int(math.floor(num_samples * train_ratio)))
            num_val = max(1, int(math.floor(num_samples * val_ratio)))
            num_test = num_samples - num_train - num_val

            # 调整样本数量，确保测试集至少有一个样本
            if num_test <= 0:
                num_test = 1
                # 从训练集中减少一个样本，确保总数不变
                num_train = max(1, num_train - 1)

            # 分配索引到对应数据集
            train_indices.extend(shuffled_indices[:num_train])
            val_indices.extend(shuffled_indices[num_train:num_train + num_val])
            test_indices.extend(shuffled_indices[num_train + num_val:])

        # Step 4: 训练集子集抽样（仅当指定了子集比例时执行）
        if 0 < self.hparams.train_subset_ratio < 1.0:
            # 构建训练集中类别到样本索引的映射
            train_class_to_indices = defaultdict(list)
            for sample_idx in train_indices:
                label = all_targets[sample_idx]
                train_class_to_indices[label].append(sample_idx)

            sampled_train_indices: List[int] = []
            subset_generator = torch.Generator().manual_seed(self.hparams.seed)

            # 按类别均匀抽样训练集子集
            for label, indices in train_class_to_indices.items():
                # 计算当前类别需要抽样的样本数量
                num_samples = max(1, int(len(indices) * self.hparams.train_subset_ratio))
                # 随机抽取指定数量的样本索引
                sampled_indices = [indices[i] for i in torch.randperm(
                    len(indices), generator=subset_generator).tolist()[:num_samples]]
                sampled_train_indices.extend(sampled_indices)

            # 使用抽样后的索引替换原始训练集索引
            train_indices = sampled_train_indices

        # === 构建数据集 ===
        base_train = Subset(
            ImageFolder(self.hparams.data_dir, transform=None),
            train_indices
        )

        if self.hparams.supcon:
            self.train_dataset = TwoCropDataset(base_train, self.train_transforms)
        else:
            self.train_dataset = Subset(
                ImageFolder(self.hparams.data_dir, self.train_transforms),
                train_indices
            )

        self.val_dataset = Subset(
            ImageFolder(self.hparams.data_dir, self.val_test_transforms),
            val_indices
        )
        self.test_dataset = Subset(
            ImageFolder(self.hparams.data_dir, self.val_test_transforms),
            test_indices    
        )


    def train_dataloader(self) -> DataLoader:
        """创建训练集数据加载器。

        Returns:
            训练集数据加载器，启用数据打乱。
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证集数据加载器。

        Returns:
            验证集数据加载器，不启用数据打乱。
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """创建测试集数据加载器。

        Returns:
            测试集数据加载器，不启用数据打乱。
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


