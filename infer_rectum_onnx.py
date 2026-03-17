import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


MODE_CHOICES = ["no_prompt", "box", "pts", "box+pts", "sota"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference on rectum NPZ folder with ONNX model and save masks as NPZ."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=MODE_CHOICES,
        help="Model mode: no_prompt / box / pts / box+pts / sota.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder containing .npz files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder to save predicted .npz masks.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size used by ONNX model.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers, e.g. CPUExecutionProvider CUDAExecutionProvider.",
    )
    return parser


def _collect_npz_files(input_dir: Path) -> List[Path]:
    files = sorted([p for p in input_dir.glob("*.npz") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")
    return files


def _safe_load_image(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        if "image" not in data:
            raise KeyError(f"Missing 'image' key in {npz_path}")
        image = data["image"]
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image in {npz_path}, got shape={image.shape}")
    return image.astype(np.float32)


def _build_sota_25d_map(npz_files: Sequence[Path]) -> Dict[Path, Tuple[Path, Path, Path]]:
    triplet_map: Dict[Path, Tuple[Path, Path, Path]] = {}
    for i, center in enumerate(npz_files):
        prev_path = npz_files[i - 1] if i > 0 else center
        next_path = npz_files[i + 1] if i + 1 < len(npz_files) else center
        triplet_map[center] = (prev_path, center, next_path)
    return triplet_map


def _prepare_model_input(
    npz_path: Path,
    mode: str,
    img_size: int,
    sota_triplets: Dict[Path, Tuple[Path, Path, Path]],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    center_img = _safe_load_image(npz_path)
    orig_h, orig_w = center_img.shape

    if mode == "sota":
        prev_path, center_path, next_path = sota_triplets[npz_path]
        ch0 = _safe_load_image(prev_path)
        ch1 = _safe_load_image(center_path)
        ch2 = _safe_load_image(next_path)
        image_3c = np.stack([ch0, ch1, ch2], axis=0)
    else:
        image_3c = np.stack([center_img, center_img, center_img], axis=0)

    resized = np.stack(
        [
            cv2.resize(image_3c[c], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            for c in range(3)
        ],
        axis=0,
    ).astype(np.float32)

    if mode == "no_prompt":
        model_input = resized
    else:
        model_input = resized * 255.0

    model_input = model_input[None, ...]
    return model_input, (orig_h, orig_w)


def _postprocess_mask(output_tensor: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    if output_tensor.ndim != 4:
        raise ValueError(f"Unexpected ONNX output shape: {output_tensor.shape}")
    pred_224 = np.argmax(output_tensor, axis=1)[0].astype(np.uint8)
    out_h, out_w = out_hw
    pred = cv2.resize(pred_224, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return pred.astype(np.uint8)


def run_inference(args: argparse.Namespace) -> None:
    model_path = Path(os.path.expanduser(args.model)).resolve()
    input_dir = Path(os.path.expanduser(args.input)).resolve()
    output_dir = Path(os.path.expanduser(args.output)).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    npz_files = _collect_npz_files(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    session = ort.InferenceSession(str(model_path), providers=args.providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    sota_triplets = _build_sota_25d_map(npz_files)

    print("Inference configuration:")
    print(args)
    print(f"Found {len(npz_files)} files in {input_dir}")

    for idx, npz_path in enumerate(npz_files, start=1):
        model_input, out_hw = _prepare_model_input(
            npz_path=npz_path,
            mode=args.mode,
            img_size=args.img_size,
            sota_triplets=sota_triplets,
        )
        output_tensor = session.run([output_name], {input_name: model_input})[0]
        pred_mask = _postprocess_mask(output_tensor, out_hw)

        out_path = output_dir / f"{npz_path.stem}.npz"
        np.savez_compressed(out_path, mask=pred_mask)

        if idx % 20 == 0 or idx == len(npz_files):
            print(f"[{idx}/{len(npz_files)}] saved: {out_path}")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
