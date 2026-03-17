import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ONNX external data (.onnx + .onnx.data) into standalone .onnx files"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("Model"),
        help="Directory containing ONNX files that may reference external tensor data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("singleonnx"),
        help="Directory to save standalone ONNX files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing standalone ONNX files in the output directory",
    )
    return parser.parse_args()


def uses_external_data(model_path: Path) -> bool:
    model = onnx.load(str(model_path), load_external_data=False)
    return any(initializer.external_data for initializer in model.graph.initializer)


def merge_model(model_path: Path, output_path: Path) -> None:
    model = onnx.load(str(model_path), load_external_data=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=False,
    )
    merged = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(merged)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    merged_count = 0
    skipped_count = 0
    for model_path in sorted(input_dir.glob("*.onnx")):
        if not uses_external_data(model_path):
            skipped_count += 1
            print(f"[skip] {model_path.name}: already standalone")
            continue

        output_path = output_dir / model_path.name
        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            print(f"[skip] {model_path.name}: {output_path} already exists")
            continue

        merge_model(model_path, output_path)
        merged_count += 1
        print(f"[ok]   {model_path.name} -> {output_path}")

    print(f"Merged {merged_count} model(s); skipped {skipped_count} model(s).")


if __name__ == "__main__":
    main()