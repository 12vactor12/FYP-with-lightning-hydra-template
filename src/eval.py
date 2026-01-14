"""评估模块

该模块是项目的评估入口，负责使用Hydra配置和Lightning框架对训练好的模型进行评估。

主要功能：
- 加载和解析Hydra配置
- 实例化数据模块、模型、回调和日志记录器
- 加载预训练模型检查点
- 执行模型评估
- 记录和返回评估指标
"""

from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
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
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# 初始化日志记录器
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """使用指定的检查点对数据模块的测试集进行评估。

    该方法被 @task_wrapper 装饰器包装，用于控制失败时的行为，
    对多轮运行和保存崩溃信息等场景非常有用。

    Args:
        cfg: Hydra 生成的配置字典，必须包含 ckpt_path

    Returns:
        一个元组，包含：
        - metric_dict: 评估指标字典
        - object_dict: 包含所有实例化对象的字典

    Raises:
        AssertionError: 如果 cfg.ckpt_path 未提供
    """
    # 确保提供了检查点路径
    assert cfg.ckpt_path, "必须提供检查点路径 (cfg.ckpt_path)"

    # 实例化数据模块
    log.info(f"实例化数据模块 <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 实例化模型
    log.info(f"实例化模型 <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 实例化日志记录器
    log.info("实例化日志记录器...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # 实例化回调函数
    log.info("实例化回调函数...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # 实例化训练器（用于评估）
    log.info(f"实例化训练器 <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 存储所有实例化对象
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    # 记录超参数
    if logger:
        log.info("记录超参数！")
        log_hyperparameters(object_dict)

    # 开始测试
    log.info("开始测试！")
    trainer.test(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=cfg.ckpt_path
    )

    # 如果需要生成预测结果，可以使用以下代码
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    # 获取评估指标
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """评估的主入口函数。

    Args:
        cfg: Hydra 生成的配置字典
    """
    # 应用额外工具函数（如请求标签、打印配置树等）
    extras(cfg)

    # 执行评估
    evaluate(cfg)


if __name__ == "__main__":
    # 执行主函数
    main()
