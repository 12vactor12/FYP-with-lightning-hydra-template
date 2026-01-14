"""兰花分类模型模块

该模块定义了用于兰花分类的LightningModule，支持多种模型架构。

主要功能：
- 实现了兰花分类的训练、验证和测试逻辑
- 支持多种评估指标（准确率、精确率、召回率、F1分数）
- 支持学习率调度
- 支持模型编译（PyTorch 2.0+）
- 自动跟踪最佳验证指标
"""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification import F1Score
from pytorch_metric_learning.losses import SupConLoss


class OrchidLitModule(LightningModule):
    """兰花分类的LightningModule，支持多种模型架构。
    
    该类继承自LightningModule，实现了完整的训练、验证和测试流程，
    支持多种评估指标和学习率调度策略。
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 19,
        supcon: bool = False,
        temperature: float = 0.07,
        supcon_probe: bool = False,

    ) -> None:
        """初始化OrchidLitModule。

        Args:
            net: 用于训练的模型实例
            optimizer: 用于训练的优化器
            scheduler: 用于训练的学习率调度器
            compile: 是否编译模型（PyTorch 2.0+）
            num_classes: 分类的类别数量，默认为19
            supcon: 是否使用对比学习(SupCon)模式，默认为False
            temperature: SupCon损失的温度参数，默认为0.07
        """
        super().__init__()

        # 保存超参数，以便在检查点中使用
        # 不记录到日志，避免日志冗余
        self.save_hyperparameters(logger=False)
        # 模型实例
        self.net = net
        self.supcon = supcon
        self.supcon_probe = supcon_probe
        # 损失函数：交叉熵损失
        if self.supcon:
            self.supcon_loss = SupConLoss(temperature=temperature)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        if not self.supcon:
            # 准确率指标
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
            
            # 精确率指标（宏平均）
            self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
            self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
            self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
            
            # 召回率指标（宏平均）
            self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
            self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
            self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
            
            # F1分数指标（宏平均）
            self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

            # 损失平均值指标
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()

            # 最佳验证指标跟踪
            self.val_acc_best = MaxMetric()  # 最佳验证准确率
            self.val_f1_best = MaxMetric()   # 最佳验证F1分数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行模型的前向传播。

        Args:
            x: 输入图像张量，形状为 [batch_size, channels, height, width]

        Returns:
            模型输出的对数概率（logits），形状为 [batch_size, num_classes]
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """训练开始时调用的Lightning钩子。
        
        主要作用：
        - 重置验证指标，避免训练前的验证检查影响指标
        - 确保最佳指标跟踪器处于初始状态
        """
        # 只有在非对比学习模式下才需要重置分类指标
        if not self.supcon:
            # Lightning默认在训练开始前执行验证步骤作为健全性检查，
            # 这里重置验证指标，避免这些检查结果影响正式训练的指标
            self.val_loss.reset()
            self.val_acc.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_f1.reset()
            self.val_acc_best.reset()
            self.val_f1_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对一批数据执行单个模型步骤。

        Args:
            batch: 包含输入图像和目标标签的元组，形状为 (images, labels)

        Returns:
            包含以下元素的元组：
            - loss: 损失张量
            - preds: 预测的类别索引，形状为 [batch_size]
            - targets: 目标类别索引，形状为 [batch_size]
        """
        # 从批次中解包图像和标签
        x, y = batch
        # 前向传播，获取对数概率
        logits = self.forward(x)
        # 计算损失, CE会自动softmax
        loss = self.loss(logits, y)
        # 获取预测类别（对数概率最大的类别）
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """对训练集中的一批数据执行单个训练步骤。

        Args:
            batch: 包含输入图像和目标标签的元组。在SupCon模式下，
                  格式为 ((x1, x2), y)，其中x1和x2是同一图像的两个增强版本。
            batch_idx: 当前批次的索引

        Returns:
            损失张量，用于反向传播
        """
        if self.supcon:
            # 在SupCon模式下，批次格式为 ((x1, x2), y)
            (augmented_image1, augmented_image2), labels = batch

            # 获取两个增强图像的特征表示
            features1 = self.net.forward_supcon(augmented_image1)
            features2 = self.net.forward_supcon(augmented_image2)

            # 合并特征和标签，SupConLoss需要形状为 [2*batch_size, feature_dim] 的特征
            combined_features = torch.cat([features1, features2], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            # 计算SupCon损失
            loss = self.supcon_loss(combined_features, combined_labels)

            # 记录SupCon损失到日志
            self.log(
                "train/supcon_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

        # 非SupCon模式下，执行标准分类训练步骤
        loss, preds, targets = self.model_step(batch)

        # 更新训练指标
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        
        # 记录训练指标到日志
        # on_step=False, on_epoch=True: 只在每个epoch结束时记录
        # prog_bar=True: 在进度条中显示
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # 返回损失，用于反向传播
        return loss

    def on_train_epoch_end(self) -> None:
        """训练epoch结束时调用的Lightning钩子。
        
        当前实现为空，可用于添加自定义的训练epoch结束逻辑。
        """
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """对验证集中的一批数据执行单个验证步骤。

        Args:
            batch: 包含输入图像和目标标签的元组
            batch_idx: 当前批次的索引
        """
        if self.supcon:
           return

        else:
            # 非SupCon模式下，执行标准分类验证步骤
            # 执行模型步骤，获取损失、预测和目标
            loss, preds, targets = self.model_step(batch)

            # 更新验证指标
            self.val_loss(loss)
            self.val_acc(preds, targets)
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1(preds, targets)
            
            # 记录验证指标到日志
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时调用的Lightning钩子。
        
        主要作用：
        - 更新最佳验证指标
        - 记录最佳验证指标到日志
        """
        # 只有在非对比学习模式下才更新最佳验证指标
        if not self.supcon:
            # 获取当前验证准确率和F1分数
            acc = self.val_acc.compute()
            f1 = self.val_f1.compute()
            
            # 更新最佳验证指标
            self.val_acc_best(acc)  # 更新最佳验证准确率
            self.val_f1_best(f1)    # 更新最佳验证F1分数
            
            # 记录最佳指标到日志
            # sync_dist=True: 在分布式训练中同步指标
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """对测试集中的一批数据执行单个测试步骤。

        Args:
            batch: 包含输入图像和目标标签的元组
            batch_idx: 当前批次的索引
        """
        # 只有在非对比学习模式下才执行测试步骤
        if not self.supcon:
            # 执行模型步骤，获取损失、预测和目标
            loss, preds, targets = self.model_step(batch)

            # 更新测试指标
            self.test_loss(loss)
            self.test_acc(preds, targets)
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
            self.test_f1(preds, targets)
            
            # 记录测试指标到日志
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """测试epoch结束时调用的Lightning钩子。
        
        当前实现为空，可用于添加自定义的测试epoch结束逻辑。
        """
        pass

    def setup(self, stage: str) -> None:
        """在训练、验证、测试或预测开始时调用的Lightning钩子。

        该钩子在使用DDP时会在每个进程上调用，适合动态构建模型或调整模型参数。

        Args:
            stage: 当前阶段，可选值为 "fit"、"validate"、"test" 或 "predict"
        """
        # 如果配置了模型编译且当前阶段为训练，则编译模型
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器。

        Returns:
            包含配置好的优化器和学习率调度器的字典
        """
        # 实例化优化器
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # 只有在非对比学习模式下才配置优化器
        if not self.supcon:
            # 如果配置了调度器，则返回优化器和调度器
            if self.hparams.scheduler is not None:
                # 实例化学习率调度器
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",  # 监控的指标
                    "interval": "epoch",    # 调度器更新的间隔单位
                    "frequency": 1,         # 调度器更新的频率
                },
            }
        
        # 否则只返回优化器
        return {"optimizer": optimizer}

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     if self.supcon_probe:
    #         """加载checkpoint时只保留backbone的权重，忽略分类头的权重。"""
    #         model_state = checkpoint.get("state_dict", {})
    #         current_state = self.state_dict()

    #         # 过滤权重：只保留backbone的权重，忽略分类头和其他组件的权重
    #         filtered_state = {}
    #         for k, v in model_state.items():
    #             # 只保留backbone的权重
    #             if k.startswith("net.backbone.") and k in current_state and v.shape == current_state[k].shape:
    #                 filtered_state[k] = v
    #             # 如果是probe训练，我们希望使用新的分类头，所以忽略原来的分类头权重
    #             elif k.startswith("net.classifier."):
    #                 continue
    #             # 忽略投影头权重（如果有的话）
    #             elif k.startswith("net.projection_head."):
    #                 continue

    #         # 用过滤后的权重更新checkpoint
    #         checkpoint["state_dict"] = filtered_state
            
    #         # 删除优化器和调度器状态，这样Lightning就会使用新的优化器和调度器
    #         # 而不是从检查点中加载
    #         if "optimizer_states" in checkpoint:
    #             del checkpoint["optimizer_states"]
    #         if "lr_schedulers" in checkpoint:
    #             del checkpoint["lr_schedulers"]
    #         if "trainer_states" in checkpoint:
    #             del checkpoint["trainer_states"]
    #     return super().on_load_checkpoint(checkpoint)

if __name__ == "__main__":
    # 测试类的实例化，确保没有语法错误
    _ = OrchidLitModule(None, None, None, None, num_classes=10)