import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from backbone import UNet as Downsample
from segment_anything import sam_model_registry


PIXEL_MEAN = (0.1364736, 0.1364736, 0.1364736)
PIXEL_STD = (0.23238614, 0.23238614, 0.23238614)
POINT_COUNT = 6
POINT_COUNT_WITH_PAD = 7

MODE_CONFIG = {
    "no_prompt": {"use_box": False, "use_points": False, "use_input_stem": False},
    "box": {"use_box": True, "use_points": False, "use_input_stem": False},
    "pts": {"use_box": False, "use_points": True, "use_input_stem": False},
    "box+pts": {"use_box": True, "use_points": True, "use_input_stem": False},
    "sota": {"use_box": True, "use_points": True, "use_input_stem": True},
}


def load_checkpoint_compat(path_or_file, map_location="cpu"):
    try:
        return torch.load(path_or_file, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path_or_file, map_location=map_location)


def _normalize_state_dict(checkpoint):
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


class ExportableUSAM(nn.Module):
    def __init__(self, mode: str, img_size: int, sam_num_classes: int):
        super().__init__()
        if mode not in MODE_CONFIG:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.img_size = img_size
        self.use_box = MODE_CONFIG[mode]["use_box"]
        self.use_points = MODE_CONFIG[mode]["use_points"]
        self.use_input_stem = MODE_CONFIG[mode]["use_input_stem"]

        self.sam = sam_model_registry["vit_b"](
            num_classes=sam_num_classes,
            img_size=img_size,
            checkpoint=None,
        )
        self.backbone = Downsample()
        if self.use_input_stem:
            self.input_stem = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 3, kernel_size=1),
            )
        else:
            self.input_stem = None

        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image, boxes=None, points=None, point_labels=None):
        image = (image - self.pixel_mean) / self.pixel_std
        if self.input_stem is not None:
            image = self.input_stem(image)

        bt_feature, skip_feature = self.backbone(image)
        image_embedding = self.sam.image_encoder(bt_feature)

        batch_size = image.shape[0]
        prompt_encoder = self.sam.prompt_encoder
        sparse_embeddings = torch.empty((batch_size, 0, prompt_encoder.embed_dim), device=image.device)
        if self.use_points:
            point_embeddings = self._embed_points_onnx(points, point_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if self.use_box:
            box_embeddings = prompt_encoder._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        dense_embeddings = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size,
            -1,
            prompt_encoder.image_embedding_size[0],
            prompt_encoder.image_embedding_size[1],
        )

        masks, _, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        masks = self.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.img_size, self.img_size],
        )
        return masks

    def _embed_points_onnx(self, points, point_labels):
        prompt_encoder = self.sam.prompt_encoder
        points = points + 0.5
        point_embedding = prompt_encoder.pe_layer.forward_with_coords(points, prompt_encoder.input_image_size)

        labels = point_labels.unsqueeze(-1)
        zeros = torch.zeros_like(point_embedding)
        not_a_point = prompt_encoder.not_a_point_embed.weight.view(1, 1, -1)
        negative_point = prompt_encoder.point_embeddings[0].weight.view(1, 1, -1)
        positive_point = prompt_encoder.point_embeddings[1].weight.view(1, 1, -1)

        point_embedding = torch.where(labels == -1, zeros, point_embedding)
        point_embedding = torch.where(labels == -1, point_embedding + not_a_point, point_embedding)
        point_embedding = torch.where(labels == 0, point_embedding + negative_point, point_embedding)
        point_embedding = torch.where(labels == 1, point_embedding + positive_point, point_embedding)
        return point_embedding


def _build_export_inputs(mode: str, img_size: int):
    image = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    boxes = torch.tensor([[32.0, 48.0, 180.0, 192.0]], dtype=torch.float32)
    points = torch.tensor(
        [[[40.0, 60.0], [56.0, 74.0], [72.0, 88.0], [88.0, 102.0], [104.0, 116.0], [120.0, 130.0]]],
        dtype=torch.float32,
    )
    point_labels = torch.ones((1, POINT_COUNT), dtype=torch.int64)
    padded_points = torch.tensor(
        [[[40.0, 60.0], [56.0, 74.0], [72.0, 88.0], [88.0, 102.0], [104.0, 116.0], [120.0, 130.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    padded_point_labels = torch.tensor([[1, 1, 1, 1, 1, 1, -1]], dtype=torch.int64)

    if mode == "no_prompt":
        return (image,), ["image"]
    if mode == "pts":
        return (image, boxes, padded_points, padded_point_labels), ["image", "boxes", "points", "point_labels"]
    return (image, boxes, points, point_labels), ["image", "boxes", "points", "point_labels"]


def convert_pth_to_onnx(
    pth_path: str,
    mode: str,
    output_path: str | None = None,
    img_size: int = 224,
    sam_num_classes: int = 3,
    opset_version: int = 18,
):
    pth_file = Path(pth_path)
    if not pth_file.exists():
        raise FileNotFoundError(f"pth file not found: {pth_file}")

    if output_path is None:
        output_path = str(pth_file.with_suffix(".onnx"))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {pth_file}")
    checkpoint = load_checkpoint_compat(pth_file, map_location="cpu")
    state_dict = _normalize_state_dict(checkpoint)

    model = ExportableUSAM(mode=mode, img_size=img_size, sam_num_classes=sam_num_classes)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"pixel_mean", "pixel_std"}
    real_missing = [key for key in missing_keys if key not in allowed_missing]
    if real_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match export model. "
            f"Missing keys: {real_missing}. Unexpected keys: {unexpected_keys}."
        )

    model.eval()
    args, input_names = _build_export_inputs(mode, img_size)
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch_size"}

    print(f"Exporting {mode} checkpoint to: {output_file}")
    torch.onnx.export(
        model,
        args,
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")
    print("Inputs:")
    for tensor in onnx_model.graph.input:
        dims = [dim.dim_value or dim.dim_param for dim in tensor.type.tensor_type.shape.dim]
        print(f"  - {tensor.name}: {dims}")
    return str(output_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert U-SAM pth checkpoint to ONNX format")
    parser.add_argument("--pth_path", type=str, required=True, help="Path to the input pth checkpoint file")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(MODE_CONFIG.keys()),
        help="Checkpoint type to export",
    )
    parser.add_argument("--output", type=str, default=None, help="Path for the output ONNX file")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--sam_num_classes", type=int, default=3, help="Number of segmentation classes")
    parser.add_argument("--opset_version", type=int, default=18, help="ONNX opset version")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    convert_pth_to_onnx(
        pth_path=cli_args.pth_path,
        mode=cli_args.mode,
        output_path=cli_args.output,
        img_size=cli_args.img_size,
        sam_num_classes=cli_args.sam_num_classes,
        opset_version=cli_args.opset_version,
    )
