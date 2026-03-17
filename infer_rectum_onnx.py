import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import zoom


POINT_INDEX = [0, 2, 4]
PTS_PER_CLASS = 10
MODE_CONFIG = {
    "no_prompt": {"use_box": False, "use_points": False, "use_adjacent": False},
    "box": {"use_box": True, "use_points": False, "use_adjacent": False},
    "pts": {"use_box": False, "use_points": True, "use_adjacent": False},
    "box+pts": {"use_box": True, "use_points": True, "use_adjacent": False},
    "sota": {"use_box": True, "use_points": True, "use_adjacent": True},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run rectum ONNX inference and save NPZ outputs")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(MODE_CONFIG.keys()),
        help="ONNX model type",
    )
    parser.add_argument("--input", type=str, required=True, help="Directory containing input NPZ files")
    parser.add_argument("--output", type=str, required=True, help="Directory for output NPZ files")
    parser.add_argument("--img_size", type=int, default=224, help="Model input size")
    return parser.parse_args()


def _sorted_npz_files(input_dir: Path):
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {input_dir}")
    return files


def _resize_image(image: np.ndarray, img_size: int) -> np.ndarray:
    height, width = image.shape
    if height == img_size and width == img_size:
        return image.astype(np.float32, copy=False)
    return zoom(image, (img_size / height, img_size / width), order=3).astype(np.float32)


def _resize_mask(mask: np.ndarray, img_size: int) -> np.ndarray:
    height, width = mask.shape
    if height == img_size and width == img_size:
        return mask.astype(np.int64, copy=False)
    return zoom(mask, (img_size / height, img_size / width), order=0).astype(np.int64)


def _build_box_prompt(label: np.ndarray, img_size: int) -> np.ndarray:
    y_indices, x_indices = np.where(label > 0)
    if len(x_indices) == 0:
        center = img_size // 2
        return np.array([[center - 1, center - 1, center + 1, center + 1]], dtype=np.float32)

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    margin = 2
    height, width = label.shape
    x_min = max(0, x_min - margin)
    x_max = min(width, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(height, y_max + margin)
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    box[[0, 2]] *= img_size / width
    box[[1, 3]] *= img_size / height
    return box[None, :]


def _sample_class_points(mask: np.ndarray, cls_idx: int) -> np.ndarray | None:
    posx, posy = np.where(mask == cls_idx)
    if len(posx) == 0:
        return None

    step = max(len(posx) // PTS_PER_CLASS, 1)
    points = []
    for index in range(PTS_PER_CLASS):
        sample_index = min(index * step, len(posx) - 1)
        points.append([posx[sample_index], posy[sample_index]])
    return np.asarray(points, dtype=np.float32)[None, :, :]


def _build_point_prompt(label: np.ndarray, img_size: int, expected_count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    resized_mask = _resize_mask(label, img_size)
    points = None
    for cls_idx in range(1, 3):
        sampled = _sample_class_points(resized_mask, cls_idx)
        if sampled is None:
            continue
        points = sampled if points is None else np.concatenate([points, sampled], axis=0)

    if points is None:
        center = np.array([[img_size // 2, img_size // 2]] * PTS_PER_CLASS, dtype=np.float32)[None, :, :]
        points = center

    points = points[None, ...]
    if points.shape[1] > 1:
        points = points[:, :, [0, 2, 4, 6, 8], :]
    else:
        points = points.reshape(1, 2, 5, 2)

    points = points[:, :, POINT_INDEX, :].reshape(1, -1, 2).astype(np.float32)
    point_labels = np.ones((1, points.shape[1]), dtype=np.int64)

    if expected_count == 7:
        padding_point = np.zeros((1, 1, 2), dtype=np.float32)
        padding_label = -np.ones((1, 1), dtype=np.int64)
        points = np.concatenate([points, padding_point], axis=1)
        point_labels = np.concatenate([point_labels, padding_label], axis=1)
    elif expected_count != points.shape[1]:
        if expected_count < points.shape[1]:
            points = points[:, :expected_count, :]
            point_labels = point_labels[:, :expected_count]
        else:
            repeat_count = expected_count - points.shape[1]
            points = np.concatenate([points, np.repeat(points[:, -1:, :], repeat_count, axis=1)], axis=1)
            point_labels = np.concatenate([point_labels, np.ones((1, repeat_count), dtype=np.int64)], axis=1)
    return points, point_labels


def _load_npz(path: Path):
    data = np.load(path)
    return data["image"].astype(np.float64), data["label"].astype(np.float64)


def _build_image_tensor(paths, index: int, mode: str, img_size: int) -> np.ndarray:
    current_image, _ = _load_npz(paths[index])

    if MODE_CONFIG[mode]["use_adjacent"]:
        prev_image, _ = _load_npz(paths[max(0, index - 1)])
        next_image, _ = _load_npz(paths[min(len(paths) - 1, index + 1)])
        channels = [prev_image, current_image, next_image]
    else:
        channels = [current_image, current_image, current_image]

    resized_channels = [_resize_image(channel, img_size) for channel in channels]
    return np.stack(resized_channels, axis=0)[None, ...].astype(np.float32)


def _build_feed(session: ort.InferenceSession, mode: str, image: np.ndarray, label: np.ndarray, img_size: int):
    input_names = {input_meta.name for input_meta in session.get_inputs()}
    needs_box = "boxes" in input_names
    needs_points = "points" in input_names or "point_labels" in input_names or "labels" in input_names

    boxes = _build_box_prompt(label, img_size) if needs_box else None
    if needs_points:
        point_count = 6
        for input_meta in session.get_inputs():
            if input_meta.name == "points" and len(input_meta.shape) > 1 and isinstance(input_meta.shape[1], int):
                point_count = input_meta.shape[1]
                break
        points, point_labels = _build_point_prompt(label, img_size, expected_count=point_count)
    else:
        points, point_labels = None, None

    feed = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        if name in {"input", "image"}:
            feed[name] = image
        elif name == "boxes":
            if boxes is None:
                raise ValueError(f"Model expects boxes input but mode={mode} does not provide it")
            feed[name] = boxes.astype(np.float32)
        elif name == "points":
            if points is None:
                raise ValueError(f"Model expects points input but mode={mode} does not provide it")
            feed[name] = points.astype(np.float32)
        elif name in {"point_labels", "labels"}:
            if point_labels is None:
                raise ValueError(f"Model expects point labels input but mode={mode} does not provide it")
            feed[name] = point_labels
        else:
            raise ValueError(f"Unsupported ONNX input name: {name}")
    return feed


def _postprocess_prediction(output: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if output.ndim == 4:
        prediction = np.argmax(output, axis=1)[0]
    elif output.ndim == 3:
        prediction = output[0]
    else:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")

    resized = cv2.resize(
        prediction.astype(np.uint8),
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.float64)


def _save_output(path: Path, image: np.ndarray, label: np.ndarray):
    np.savez_compressed(
        path,
        image=np.asfortranarray(image.astype(np.float64)),
        label=np.asfortranarray(label.astype(np.float64)),
    )


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_files = _sorted_npz_files(input_dir)

    for index, npz_path in enumerate(input_files):
        image_512, label_512 = _load_npz(npz_path)
        image_tensor = _build_image_tensor(input_files, index, args.mode, args.img_size)
        feed = _build_feed(session, args.mode, image_tensor, label_512, args.img_size)
        logits = session.run(None, feed)[0]
        pred_label = _postprocess_prediction(logits, label_512.shape)
        _save_output(output_dir / npz_path.name, image_512, pred_label)

    print(f"Saved {len(input_files)} NPZ files to {output_dir}")


if __name__ == "__main__":
    main()