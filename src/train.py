"""训练模块

该模块是项目的训练入口，负责使用Hydra配置和Lightning框架进行模型训练、验证和测试。

主要功能：
- 加载和解析Hydra配置
- 实例化数据模块、模型、回调和日志记录器
- 执行模型训练和测试
- 记录和返回训练指标
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# 设置项目根目录，确保导入路径正确
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# rootutils.setup_root 等效于：
# - 将项目根目录添加到 PYTHONPATH
# - 设置 PROJECT_ROOT 环境变量（用于配置文件中的路径）
# - 从根目录的 .env 文件加载环境变量

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# 初始化日志记录器
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """训练模型，并可选地在测试集上进行评估。

    该方法被 @task_wrapper 装饰器包装，用于控制失败时的行为，
    对多轮运行和保存崩溃信息等场景非常有用。

    Args:
        cfg: Hydra 生成的配置字典

    Returns:
        一个元组，包含：
        - metric_dict: 合并了训练和测试指标的字典
        - object_dict: 包含所有实例化对象的字典
    """
    # 设置随机种子，确保实验可复现
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 实例化数据模块
    log.info(f"实例化数据模块 <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 实例化模型
    log.info(f"实例化模型 <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # 手动加载权重
    if cfg.get("supcon_probe_first") and cfg.get("ckpt_path"):
        # 加载权重，设置strict=False
        model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'], strict=False)
    # 实例化回调函数
    log.info("实例化回调函数...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))


    # 实例化日志记录器
    log.info("实例化日志记录器...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # 实例化训练器
    log.info(f"实例化训练器 <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # 存储所有实例化对象
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # 记录超参数
    if logger:
        log.info("记录超参数！")
        log_hyperparameters(object_dict)

    # 执行训练
    if cfg.get("train"):
        log.info("开始训练！")
        trainer.fit(
            model=model, 
            datamodule=datamodule, 
            ckpt_path=cfg.get("ckpt_path") if not cfg.get("supcon_probe_first") else None
        )

    # 获取训练指标
    train_metrics = trainer.callback_metrics

    # 执行测试
    if cfg.get("test"):
        log.info("开始测试！")
        # 获取最佳检查点路径
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("未找到最佳检查点！使用当前权重进行测试...")
            ckpt_path = None
        # 使用最佳检查点或当前权重进行测试
        trainer.test(
            model=model, 
            datamodule=datamodule, 
            ckpt_path=ckpt_path
        )
        log.info(f"最佳检查点路径: {ckpt_path}")

    # 获取测试指标
    test_metrics = trainer.callback_metrics

    # 合并训练和测试指标
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """训练的主入口函数。

    Args:
        cfg: Hydra 生成的配置字典

    Returns:
        用于 Hydra 超参数优化的优化指标值（可选）
    """
    # 应用额外工具函数（如请求标签、打印配置树等）
    extras(cfg)

    # 训练模型
    metric_dict, _ = train(cfg)

    # 安全地获取优化指标值，用于 Hydra 超参数优化
    metric_value = get_metric_value(
        metric_dict=metric_dict, 
        metric_name=cfg.get("optimized_metric")
    )

    # 返回优化指标
    return metric_value


if __name__ == "__main__":
    # 执行主函数
    main()
