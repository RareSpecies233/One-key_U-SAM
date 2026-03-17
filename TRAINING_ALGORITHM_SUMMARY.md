--------------------------

- 修复了 pth 转 onnx 脚本 convert_pth_to_onnx.py，不再把 prompt 路径导丢，并按模式区分普通 U-SAM 与带 input_stem 的 sota 导出骨架。
- 新增 infer_rectum_onnx.py，支持 --model、--mode、--input、--output，兼容旧 no_prompt 单输入 onnx 和新 prompt 多输入 onnx。
- 输出 npz 已统一为 image 和 label 两个键，二者均为 (512, 512)、float64、Fortran-order。
- 重新导出了 4 个 pth 对应的 onnx：box、pts、box+pts、sota；no_prompt 因仓库内无对应 pth，继续使用现有 onnx。
- 使用 /Users/rarespecies/Documents/folder/temp/17105001 完成五模型推理测试，输出目录为 /Users/rarespecies/Documents/folder/temp/17105001_outputs_onnx_fixed。
- 快速前景 Dice 结果：no_prompt 0.779523，box 0.169879，pts 0.841783，box+pts 0.874274，sota 0.615414。