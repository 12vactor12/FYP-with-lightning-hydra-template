# src/data/my_dataset_datamodule.py
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Optional

class MyDatasetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/my_dataset/",
        batch_size: int = 32,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 数据变换
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
        
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        # 数据集已经存在，不需要下载
        pass
        
    def setup(self, stage: Optional[str] = None):
        # 加载完整数据集
        if self.dataset is None:
            self.dataset = ImageFolder(
                root=self.hparams.data_dir,
                transform=self.train_transforms
            )
        
        # 划分数据集
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            total_size = len(self.dataset)
            train_size = int(total_size * self.hparams.train_val_test_split[0])
            val_size = int(total_size * self.hparams.train_val_test_split[1])
            test_size = total_size - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size]
            )
            
            # 设置验证集和测试集的变换
            self.val_dataset.dataset.transform = self.val_test_transforms
            self.test_dataset.dataset.transform = self.val_test_transforms
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )