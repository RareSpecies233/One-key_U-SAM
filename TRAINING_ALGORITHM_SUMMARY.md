--------------------------

- 修复了 pth 转 onnx 脚本 convert_pth_to_onnx.py，不再把 prompt 路径导丢，并按模式区分普通 U-SAM 与带 input_stem 的 sota 导出骨架。
- 新增 infer_rectum_onnx.py，支持 --model、--mode、--input、--output，兼容旧 no_prompt 单输入 onnx 和新 prompt 多输入 onnx。
- 输出 npz 已统一为 image 和 label 两个键，二者均为 (512, 512)、float64、Fortran-order。
- 重新导出了 4 个 pth 对应的 onnx：box、pts、box+pts、sota；no_prompt 因仓库内无对应 pth，继续使用现有 onnx。
- 使用 /Users/rarespecies/Documents/folder/temp/17105001 完成五模型推理测试，输出目录为 /Users/rarespecies/Documents/folder/temp/17105001_outputs_onnx_fixed。
- 快速前景 Dice 结果：no_prompt 0.779523，box 0.169879，pts 0.841783，box+pts 0.874274，sota 0.615414。

--------------------------

- 新增 merge_external_onnx.py，用于把 Model 下带 .onnx.data 的 4 个模型合并成单文件 onnx，并输出到 singleonnx。
- 在 cppInfer 下新增 infer_rectum_onnx.cpp 和 buildmacos.sh，支持 --model、--mode、--input、--output、--img_size 参数，在 macOS 上使用 ONNX Runtime + cnpy 进行批量推理。
- 扩展 cppInfer/cnpy/cnpy.h，使 C++ 推理输出的 npz 可以正确保存为 Fortran-order，并保持 image、label 两个键均为 float64、(512, 512)。
- 已成功编译出 cppInfer/infer_rectum_onnx，并使用 /Users/rarespecies/Documents/folder/temp/17105001 对 no_prompt、box、pts、box+pts、sota 五种模型完成测试。
- C++ 推理输出目录为 /Users/rarespecies/Documents/folder/temp/17105001_cpp_outputs，各模式均生成 47 个 npz，抽样检查均为 image/label、float64、Fortran-order。
- 快速前景 Dice 结果：no_prompt 0.817797，box 0.167648，pts 0.858401，box+pts 0.874218，sota 0.602565。