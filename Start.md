

# U-SAM 项目快速上手指南

U-SAM (Universal Segment Anything Model) 是一个基于 Segment Anything Model (SAM) 的通用分割模型，适用于医学图像分割等任务。

## 快速开始

### 1. 环境设置

确保你有 Python 3.8+ 和 pip 安装。

```bash
# 克隆项目（如果需要）
# git clone <repository-url>
# cd One-key_U-SAM

# 安装依赖
pip install -r requirements.txt
```

或者使用 uv（推荐）：

```bash
uv pip install -r requirements.txt
```

### 2. 数据准备

将数据集放在指定目录下。数据集应包含：
- 训练/测试图像和标签（NPZ 格式）
- bbox CSV 文件

例如，对于 rectum 数据集：
```
data_root/
├── train/
│   ├── train_bbox.csv
│   └── train_npz/
│       ├── image1.npz
│       └── ...
└── test/
    ├── test_bbox.csv
    └── test_npz/
        ├── image1.npz
        └── ...
```

### 3. 运行训练

基本训练命令：

```bash
python u-sam.py --epochs 100 --batch_size 24 --dataset rectum --data_root /path/to/data
```

### 4. 运行推理

使用训练好的模型进行推理：

```bash
python infer_pth.py --model_path exp/U-SAM-Rectum/checkpoint.pth --input_image /path/to/image --output_dir results/
```

## u-sam.py 参数说明

本文整理 u-sam.py 的附加参数可选项、默认值与推荐值。

```
python .\u-sam.py  --epochs 100 --batch_size 24 --dataset rectum --data_root C:\Users\zheng\Desktop\Folder\MedPjt\DataV6
```

## SAM 相关

| 参数 | 可选项 | 默认值 | 推荐值 | 说明 |
| --- | --- | --- | --- | --- |
| --prompt_mode | 0, 1, 2, 3 | 0 | 0 | 0: 无 box/pts; 1: gt boxes; 2: gt pts; 3: gt boxes+pts |
| --warmup | 仅开关 | 关闭 | 关闭 | 训练初期 warmup |
| --lr | 任意浮点 | 1e-3 | 1e-3 | 主干以外参数学习率 |
| --lr_vit | 任意浮点 | 1e-4 | 1e-4 | image encoder 学习率 |
| --lr_backbone | 任意浮点 | 1e-4 | 1e-4 | 下采样 backbone 学习率 |
| --batch_size | 任意整数 | 24 | 24 | 根据显存可调 |
| --weight_decay | 任意浮点 | 1e-4 | 1e-4 | 权重衰减 |
| --epochs | 任意整数 | 100 | 100 | 训练轮数 |
| --clip_max_norm | 任意浮点 | 0.1 | 0.1 | 梯度裁剪阈值 |

## 数据集相关

| 参数 | 可选项 | 默认值 | 推荐值 | 说明 |
| --- | --- | --- | --- | --- |
| --img_size | 任意整数 | 224 | 224 | 输入尺寸 |
| --dataset | rectum, word | rectum | rectum | 数据集类型 |
| --data_root | 任意路径 | 空 | 必填 | 数据集根目录 |

## 运行配置

| 参数 | 可选项 | 默认值 | 推荐值 | 说明 |
| --- | --- | --- | --- | --- |
| --output_dir | 任意路径 | ./exp/U-SAM-Rectum | ./exp/U-SAM-Rectum | 输出目录 |
| --device | cuda, cpu | cuda | cuda | 训练设备 |
| --seed | 任意整数 | 202307 | 202307 | 随机种子 |
| --resume | 任意路径 | 空 | 空 | 断点恢复权重 |
| --start_epoch | 任意整数 | 0 | 0 | 起始 epoch |
| --eval | 仅开关 | 关闭 | 关闭 | 仅评估 |
| --num_workers | 任意整数 | 2 | 2 | DataLoader worker 数 |

## 分布式相关

| 参数 | 可选项 | 默认值 | 推荐值 | 说明 |
| --- | --- | --- | --- | --- |
| --world_size | 任意整数 | 1 | 1 | 进程数 |
| --dist_url | 任意字符串 | env:// | env:// | 分布式初始化地址 |

## 故障排除

- 如果遇到 "Sample larger than population" 错误，请检查数据集中的掩码是否完整。
- 确保 CUDA 版本与 PyTorch 兼容。
- 数据路径请使用绝对路径。
