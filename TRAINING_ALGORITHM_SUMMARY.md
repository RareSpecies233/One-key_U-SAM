---
### 2026-03-17 会话总结
- 复用并确认了批量 ONNX 推理脚本 [infer_onnx_batch.py](infer_onnx_batch.py)，其参数已满足：`--model`、`--mode`（`no_prompt`/`box`/`pts`/`box+pts`/`sota`）、`--input`、`--output`。
- 在你指定的 `uv` 项目环境（解释器路径：`/Users/rarespecies/Documents/folder/One-key_U-SAM/.venv/bin/python`）执行了 5 种模型推理测试。
- 处理了 `pts`/`box+pts` 模型 ONNX 外部数据文件命名不一致问题：在脚本中新增自动匹配并补齐缺失 `.onnx.data` 引用的逻辑，避免 ONNXRuntime 初始化失败。
- 使用输入目录：`/Users/rarespecies/Documents/folder/temp/17105001`，五种模式输出到：`/Users/rarespecies/Documents/folder/temp/17105001_outputs_v2/<mode>`。
- 最终结果：5 种模式均成功完成，且每个模式输出 `47` 个 `.npz` 文件。

---
### 2026-03-17 修复总结（空掩码问题）
- 你反馈 `no_prompt` 以外模式输出“无信息（全背景）”，已复现并定位。
- 根因：5 个 ONNX 模型的输入像素值域不一致，`no_prompt` 适配 `[0,1]`，而 `box/pts/box+pts/sota` 更适配 `[0,255]`。
- 已修改 [infer_onnx_batch.py](infer_onnx_batch.py)：
	- 新增参数 `--pixel_scale`，支持 `auto|01|255`；默认 `auto`。
	- `auto` 策略：`no_prompt -> [0,1]`，其余模式 `-> [0,255]`。
- 修复后在同一输入目录重新测试输出到 `/Users/rarespecies/Documents/folder/temp/17105001_outputs_v3`，统计如下：
	- `no_prompt`: 47 文件，46 个切片非空
	- `box`: 47 文件，47 个切片非空
	- `pts`: 47 文件，24 个切片非空
	- `box+pts`: 47 文件，47 个切片非空
	- `sota`: 47 文件，47 个切片非空
