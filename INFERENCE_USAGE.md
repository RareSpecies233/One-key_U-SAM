# 推理脚本使用说明

## 环境激活（UV）

请先在项目根目录激活 UV 环境，然后再运行脚本。下面是示例命令：

```bash
uv venv
uv pip install -r requirements.txt
uv run python --version
```

> 如果你已有现成的 UV 环境，请将上面的命令替换为你本地的实际激活流程。

## PyTorch 推理（infer_pth.py）

脚本位置：Rare-tools/infer_pth.py

### 必填参数

- `npz`：输入 .npz 文件路径
- `--model`：.pth 权重文件路径

### 可选参数

- `--config`：推理配置文件 inference_params.json 的路径
- `--output`：输出 .npz 文件路径（不填则输出到输入文件同目录）

### 示例

```bash
uv run python Rare-tools/infer_pth.py data/example.npz --model checkpoints/best.pth
```

```bash
uv run python Rare-tools/infer_pth.py data/example.npz --model checkpoints/best.pth --config configs/inference_params.json --output outputs/example_pred.npz
```

## ONNX 推理（infer_onnx.py）

脚本位置：Rare-tools/infer_onnx.py

### 必填参数

- `--onnx`：ONNX 模型路径
- `--npz`：输入 .npz 文件路径

### 可选参数

- `--config`：推理配置文件 inference_params.json 的路径
- `--output`：输出 .npz 文件路径（不填则输出到输入文件同目录）

### 示例

```bash
uv run python Rare-tools/infer_onnx.py --onnx checkpoints/usam.onnx --npz data/example.npz
```

```bash
uv run python Rare-tools/infer_onnx.py --onnx checkpoints/usam.onnx --npz data/example.npz --config configs/inference_params.json --output outputs/example_pred_onnx.npz
```

## 配置文件查找规则

如果不传 `--config`，脚本会依次尝试以下位置：

1. 与输入 .npz 同目录的 inference_params.json
2. Rare-tools 目录下的 inference_params.json

如果仍未找到，会抛出错误并提示显式传入 `--config`。
