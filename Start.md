# u-sam.py 参数说明

本文整理 u-sam.py 的附加参数可选项、默认值与推荐值。

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
