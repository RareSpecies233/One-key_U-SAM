import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset.rectum_dataloader import RectumDataloader
import util.misc as utils


def _load_sam_and_utils() -> Tuple[torch.nn.Module, object]:
    """
    Dynamically load SAM and load_checkpoint_compat from u-sam.py.
    We cannot `import u-sam` directly because of the hyphen in the filename.
    """
    models_file = Path(__file__).resolve().parent / "u-sam.py"
    spec = importlib.util.spec_from_file_location("u_sam_module", models_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {models_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]

    if not hasattr(module, "SAM"):
        raise AttributeError(f"'SAM' not found in {models_file}")
    if not hasattr(module, "load_checkpoint_compat"):
        raise AttributeError(f"'load_checkpoint_compat' not found in {models_file}")

    return module.SAM, module.load_checkpoint_compat  # type: ignore[attr-defined]


SAM, load_checkpoint_compat = _load_sam_and_utils()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference on rectum test set with new U-SAM and save masks as NPZ."
    )

    # dataset / dataloader
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/alaph/sunchen/USAM+/DataV6",
        help="Root directory of DataV6 (containing train/test folders).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size (must match training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    # model / checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Path to best_*.pth checkpoint. "
            "If empty, the latest best_*.pth under exp/U-SAM-Rectum will be used."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Must match training: 0=no prompt, 1=gt boxes, 2=gt pts, 3=gt boxes+pts.",
    )

    # keep flag for compatibility with training, although RectumDataloader itself
    # does not take use_2_5d in its __init__ in the current implementation.
    parser.add_argument(
        "--use_2_5d",
        action="store_true",
        help="Enable 2.5D mode (three adjacent slices as channels). Must match training.",
    )

    # runtime
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on.",
    )

    # output
    parser.add_argument(
        "--output_npz_dir",
        type=str,
        default="/home/alaph/sunchen/USAM+/DataV6/test/test_pred_npz",
        help="Directory to save predicted masks as .npz files.",
    )

    return parser


def _select_latest_best_checkpoint(base_dir: Path) -> Optional[Path]:
    """
    Search for the latest best_*.pth checkpoint under exp/U-SAM-Rectum.
    Returns None if not found.
    """
    search_root = base_dir / "exp" / "U-SAM-Rectum"
    if not search_root.exists():
        return None

    candidates = list(search_root.rglob("best_*.pth"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _set_prompt_flags(args: argparse.Namespace) -> None:
    """
    Match the prompt_mode -> flags logic used in u-sam.py::main.
    """
    if args.prompt_mode == 0:
        args.use_gt_box = False
        args.use_gt_pts = False
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
    elif args.prompt_mode == 1:
        args.use_gt_box = True
        args.use_gt_pts = False
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
    elif args.prompt_mode == 2:
        args.use_gt_box = False
        args.use_gt_pts = True
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
    elif args.prompt_mode == 3:
        args.use_gt_box = True
        args.use_gt_pts = True
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
    else:
        raise ValueError(f"Unsupported prompt_mode={args.prompt_mode}")


def build_sam_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Build the SAM model using the new implementation from u-sam.py,
    keeping critical parameters consistent with training.
    """
    # dataset-specific settings (rectum)
    args.sam_num_classes = 3
    args.img_size = int(args.img_size)

    _set_prompt_flags(args)

    # loss weight parameters: needed by SAM.__init__, but only used during training
    if not hasattr(args, "dice_weight"):
        args.dice_weight = 0.6
    if not hasattr(args, "boundary_weight"):
        # Training script sets boundary_loss weight to 0; inference does not use it,
        # but we keep a sensible default for completeness.
        args.boundary_weight = 0.0

    model = SAM(args)

    # normalization statistics as in u-sam.py (training)
    pixel_mean = (0.1364736, 0.1364736, 0.1364736)
    pixel_std = (0.23238614, 0.23238614, 0.23238614)
    model.pixel_mean = pixel_mean
    model.pixel_std = pixel_std

    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    Load checkpoint using the same compatibility helper as in u-sam.py.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    with open(ckpt_path, "rb") as f:
        checkpoint = load_checkpoint_compat(f, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def build_test_loader(args: argparse.Namespace) -> DataLoader:
    """
    Build the rectum test loader. Signature matches current RectumDataloader.
    """
    dataset_test = RectumDataloader(
        root_dir=args.data_root,
        mode="test",
        imgsize=(args.img_size, args.img_size),
    )
    sampler = SequentialSampler(dataset_test)
    data_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        drop_last=False,
    )
    return data_loader


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # resolve checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        maybe_ckpt = _select_latest_best_checkpoint(Path(__file__).resolve().parent)
        if maybe_ckpt is None:
            raise FileNotFoundError(
                "No --checkpoint provided and no best_*.pth found under exp/U-SAM-Rectum."
            )
        ckpt_path = str(maybe_ckpt)
        print(f"Using latest best checkpoint: {ckpt_path}")
    else:
        print(f"Using specified checkpoint: {ckpt_path}")

    # build model and dataloader
    model = build_sam_model(args).to(device)
    model.eval()
    load_checkpoint(model, ckpt_path, device=device)

    data_loader = build_test_loader(args)
    dataset_test = data_loader.dataset  # type: ignore[attr-defined]

    # prepare output dir
    os.makedirs(args.output_npz_dir, exist_ok=True)

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Infer:"

    with torch.no_grad():
        for samples, targets in metric_logger.log_every(data_loader, print_freq=50, header=header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass (eval mode returns masks and metrics; we only need masks)
            masks, _, _, _, _ = model(samples, targets)  # [B, H, W]

            masks_np = masks.detach().cpu().numpy().astype(np.uint8)

            for i, target in enumerate(targets):
                idx_tensor = target["id"].cpu().item()
                filename = dataset_test.csv.iloc[idx_tensor, 0]  # type: ignore[attr-defined]

                pred_mask = masks_np[i]
                out_path = os.path.join(args.output_npz_dir, f"{filename}.npz")
                np.savez_compressed(out_path, mask=pred_mask)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if not args.data_root:
        raise ValueError("--data_root must be provided.")

    print("Inference configuration:")
    print(args)

    run_inference(args)


if __name__ == "__main__":
    main()

