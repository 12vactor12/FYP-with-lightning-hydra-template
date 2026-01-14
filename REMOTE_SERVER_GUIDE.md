# 远程服务器运行指南

本指南详细说明如何在远程服务器上配置环境、安装并运行本项目。

## 1. 连接到远程服务器

使用 SSH 连接到远程服务器：

```bash
ssh username@server_ip
```

其中：
- `username` 是您在远程服务器上的用户名
- `server_ip` 是远程服务器的 IP 地址

## 2. 安装必要的基础依赖

### 2.1 安装 Git

```bash
# Ubuntu/Debian 系统
sudo apt update
sudo apt install -y git

# CentOS/RHEL 系统
sudo yum update
sudo yum install -y git

# 验证安装
git --version
```

### 2.2 安装 Miniconda

```bash
# 下载 Miniconda 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 运行安装脚本
bash Miniconda3-latest-Linux-x86_64.sh

# 按照提示完成安装，最后选择 "yes" 初始化 Miniconda

# 重新加载 .bashrc 文件
source ~/.bashrc

# 验证安装
conda --version
```

## 3. 克隆项目代码

```bash
# 克隆项目代码
git clone https://github.com/your-username/your-project-name.git

# 进入项目目录
cd your-project-name
```

## 4. 创建并激活 Conda 环境

```bash
# 使用 environment.yaml 创建环境
conda env create -f environment.yaml

# 激活环境
conda activate myenv

# 验证环境激活
conda info --envs  # 查看所有环境，当前激活的环境会有 "*" 标记
```

## 5. 安装项目依赖

```bash
# 安装项目依赖
pip install -e .

# 验证依赖安装
pip list | grep -E "torch|lightning|hydra"
```

## 6. 运行训练或评估脚本

### 6.1 运行训练脚本

```bash
# 使用默认配置运行训练
python src/train.py

# 或者使用特定的实验配置
python src/train.py +experiment=vit_base_16

# 或者使用 Makefile 中的命令
make train
```

### 6.2 运行评估脚本

```bash
# 使用默认配置和指定的检查点运行评估
python src/eval.py ckpt_path=/path/to/your/checkpoint.ckpt

# 或者使用特定的实验配置
python src/eval.py +experiment=vit_base_16 ckpt_path=/path/to/your/checkpoint.ckpt
```

### 6.3 使用 Hydra 覆盖配置参数

```bash
# 覆盖学习率和批次大小
python src/train.py model.optimizer.lr=0.0001 data.batch_size=32

# 启用特定的回调
python src/train.py +callbacks=early_stopping
```

## 7. 使用 tmux 或 screen 进行后台运行

为了防止 SSH 连接断开导致训练中断，建议使用 tmux 或 screen 进行后台运行：

### 7.1 使用 tmux

```bash
# 安装 tmux
sudo apt install -y tmux  # Ubuntu/Debian
sudo yum install -y tmux  # CentOS/RHEL

# 创建一个新的 tmux 会话
tmux new -s training-session

# 在 tmux 会话中运行训练脚本
python src/train.py

# 按下 Ctrl+B，然后按下 D 退出 tmux 会话（训练将在后台继续）

# 重新连接到 tmux 会话
tmux attach -t training-session

# 查看所有 tmux 会话
tmux ls
```

### 7.2 使用 screen

```bash
# 安装 screen
sudo apt install -y screen  # Ubuntu/Debian
sudo yum install -y screen  # CentOS/RHEL

# 创建一个新的 screen 会话
screen -S training-session

# 在 screen 会话中运行训练脚本
python src/train.py

# 按下 Ctrl+A，然后按下 D 退出 screen 会话（训练将在后台继续）

# 重新连接到 screen 会话
screen -r training-session

# 查看所有 screen 会话
screen -ls
```

## 8. 查看训练日志

### 8.1 实时查看日志

```bash
# 使用 tail 命令实时查看日志文件
tail -f logs/train/runs/$(ls -t logs/train/runs/ | head -n 1)/train.log
```

### 8.2 使用 TensorBoard 查看训练指标

```bash
# 在远程服务器上启动 TensorBoard
tensorboard --logdir logs/tensorboard --port 6006 --bind_all
```

然后在本地浏览器中访问：
```
http://server_ip:6006
```

## 9. 常见问题及解决方案

### 9.1 CUDA 版本不匹配

如果遇到 CUDA 版本不匹配的问题，可以尝试：

```bash
# 安装特定版本的 PyTorch 和 CUDA
conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 9.2 内存不足

如果遇到内存不足的问题，可以尝试：

1. 减小批次大小：`python src/train.py data.batch_size=16`
2. 使用梯度累积：`python src/train.py trainer.accumulate_grad_batches=4`
3. 使用混合精度训练：`python src/train.py trainer.precision=16`

### 9.3 权限问题

如果遇到权限问题，可以尝试：

```bash
# 更改文件权限
chmod +x src/train.py src/eval.py
```

## 10. 项目结构说明

```
├── configs/           # Hydra 配置文件
├── src/               # 源代码
│   ├── data/          # 数据处理模块
│   ├── models/        # 模型定义
│   ├── utils/         # 工具函数
│   ├── visualization/ # 可视化脚本
│   ├── train.py       # 训练入口
│   └── eval.py        # 评估入口
├── tests/             # 测试代码
├── environment.yaml   # Conda 环境配置
├── pyproject.toml     # 项目配置
└── Makefile           # 常用命令
```

## 11. 参考文档

- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [Hydra 文档](https://hydra.cc/docs/intro/)
- [Conda 文档](https://conda.io/projects/conda/en/latest/user-guide/index.html)
- [Git 文档](https://git-scm.com/doc)
