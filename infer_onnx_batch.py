from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F


MODE_TO_DEFAULT_MODEL = {
    "no_prompt": "no_prompt_best_0.677767_0.513901.onnx",
    "box": "box_best_0.399320_0.250015.onnx",
    "pts": "pts_best_0.710546_0.552191.onnx",
    "box+pts": "box+ptsbest_0.729687_0.575057.onnx",
    "sota": "sotabest_0.731092_0.576609.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for U-SAM ONNX models")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to ONNX model. If not set, model is resolved from --mode under Model/.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["no_prompt", "box", "pts", "box+pts", "sota"],
        help="Model type: no_prompt | box | pts | box+pts | sota (2.5D)",
    )
    parser.add_argument("--input", required=True, help="Input directory that contains .npz files")
    parser.add_argument("--output", required=True, help="Output directory for predicted .npz files")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Model input image size (default: 224)",
    )
    return parser.parse_args()


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    max_val = float(np.max(image)) if image.size else 0.0
    if max_val > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def resize_2d(image: np.ndarray, out_size: int) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return tensor[0, 0].cpu().numpy()


def resize_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(out_h, out_w), mode="nearest")
    return tensor[0, 0].round().to(torch.int64).cpu().numpy()


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def build_mode_input(
    mode: str,
    all_npz_paths: List[Path],
    cur_idx: int,
    image_key: str,
    img_size: int,
) -> np.ndarray:
    if mode == "sota":
        left_idx = max(0, cur_idx - 1)
        right_idx = min(len(all_npz_paths) - 1, cur_idx + 1)

        left = normalize_image(load_npz(all_npz_paths[left_idx])[image_key])
        mid = normalize_image(load_npz(all_npz_paths[cur_idx])[image_key])
        right = normalize_image(load_npz(all_npz_paths[right_idx])[image_key])

        left = resize_2d(left, img_size)
        mid = resize_2d(mid, img_size)
        right = resize_2d(right, img_size)

        stacked = np.stack([left, mid, right], axis=0)
    else:
        image = normalize_image(load_npz(all_npz_paths[cur_idx])[image_key])
        image = resize_2d(image, img_size)
        stacked = np.stack([image, image, image], axis=0)

    return stacked[np.newaxis, ...].astype(np.float32)


def resolve_model_path(mode: str, model_arg: str | None) -> Path:
    if model_arg:
        model_path = Path(model_arg)
    else:
        repo_root = Path(__file__).resolve().parent
        model_path = repo_root / "Model" / MODE_TO_DEFAULT_MODEL[mode]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def _pick_external_data_candidate(expected_name: str, candidates: List[Path]) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    expected_tail = expected_name.replace("best_", "")
    if expected_tail:
        for cand in candidates:
            if expected_tail in cand.name:
                return cand

    metric_match = re.search(r"\d+\.\d+_\d+\.\d+", expected_name)
    if metric_match:
        metric = metric_match.group(0)
        for cand in candidates:
            if metric in cand.name:
                return cand

    return None


def ensure_external_data_files(model_path: Path) -> None:
    # Some exported ONNX models depend on external *.onnx.data files with inconsistent names.
    # This creates a best-effort alias in the model directory when the referenced filename is missing.
    raw = model_path.read_bytes()
    matches = re.findall(rb"([A-Za-z0-9_+\-.]+\.onnx\.data)", raw)
    expected_names = {m.decode("utf-8", errors="ignore") for m in matches}
    if not expected_names:
        return

    model_dir = model_path.parent
    candidates = list(model_dir.glob("*.onnx.data"))
    for expected_name in sorted(expected_names):
        expected_path = model_dir / expected_name
        if expected_path.exists():
            continue

        candidate = _pick_external_data_candidate(expected_name, candidates)
        if candidate is None:
            continue

        try:
            expected_path.symlink_to(candidate.name)
        except OSError:
            shutil.copy2(candidate, expected_path)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid --input directory: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {input_dir}")

    model_path = resolve_model_path(args.mode, args.model)

    ensure_external_data_files(model_path)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    success = 0
    for idx, npz_path in enumerate(npz_files):
        sample = load_npz(npz_path)
        if "image" not in sample:
            raise KeyError(f"Key 'image' not found in {npz_path}")

        image = sample["image"]
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

        input_tensor = build_mode_input(
            mode=args.mode,
            all_npz_paths=npz_files,
            cur_idx=idx,
            image_key="image",
            img_size=int(args.img_size),
        )

        outputs = sess.run(None, {input_name: input_tensor})
        if not outputs:
            raise RuntimeError(f"ONNX output is empty for file: {npz_path}")

        logits = outputs[0]
        pred = np.argmax(logits, axis=1)[0]
        pred = resize_mask(pred, orig_h, orig_w)

        out_data = dict(sample)
        if "label" in out_data:
            pred = pred.astype(out_data["label"].dtype, copy=False)
            out_data["label"] = pred
        else:
            out_data["pred"] = pred

        out_path = output_dir / npz_path.name
        np.savez(out_path, **out_data)
        success += 1

    print(f"Mode: {args.mode}")
    print(f"Model: {model_path}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Processed: {success}/{len(npz_files)}")


if __name__ == "__main__":
    main()
