#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python train_stage1_qwen_adapter.py \
  --jsonl data/stage1_contrastive/annotations.jsonl \
  --root_dir data/stage1_contrastive \
  --output_dir runs/stage1_qwen3vl_vit \
  --model_name /data/hyz/workspace/hf/Qwen3-VL-4B-Instruct \
  --batch_size 32 \
  --epochs 5 \
  --lr 1e-4 \
  --margin 0.3 \
  --workers 8 \
  --bf16 \
  --grad_checkpoint
'''


import argparse
import json
import os
import sys
import gc
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
)

# -----------------------------------------------------------------------------
# System Optimization: TF32 for Ampere+ GPUs
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TripletClockDataset(Dataset):
    """
    Qwen2.5-VL 视觉塔期望输入是 patchified 的 pixel_values + image_grid_thw。
    这里直接用 AutoImageProcessor 产出该格式，避免把 [B,3,H,W] 错误喂进去导致 1280/640 mismatch。
    """
    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,
        model_name: str,
        resolution: int = 0,
        triplet_type_filter: Optional[str] = None,
        use_fast_processor: Optional[bool] = None,
    ):
        print(f"Loading index from {jsonl_path}...")
        self.rows = _load_jsonl(jsonl_path)
        if triplet_type_filter:
            original_len = len(self.rows)
            self.rows = [
                r for r in self.rows
                if r.get("meta", {}).get("triplet_type") == triplet_type_filter
            ]
            print(f"Filtered dataset from {original_len} to {len(self.rows)} items (type={triplet_type_filter})")

        self.root_dir = root_dir

        kwargs = {"trust_remote_code": True}
        if use_fast_processor is not None:
            kwargs["use_fast"] = use_fast_processor

        self.processor = AutoImageProcessor.from_pretrained(model_name, **kwargs)

        # 可选：如果你确实想强行改分辨率，尽量温和处理（不同 processor 字段可能不一样）
        if resolution and hasattr(self.processor, "size") and isinstance(getattr(self.processor, "size"), dict):
            try:
                self.processor.size["height"] = resolution
                self.processor.size["width"] = resolution
                print(f"[Dataset] Override processor.size -> {self.processor.size}")
            except Exception:
                pass

    def __len__(self):
        return len(self.rows)

    def _load_pil(self, rel_path: str) -> Image.Image:
        path = os.path.join(self.root_dir, rel_path)
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            # 兜底：返回一张黑图，避免 crash
            return Image.new("RGB", (224, 224), (0, 0, 0))

    @torch.no_grad()
    def _encode_vision_inputs(self, pil_img: Image.Image):
        out = self.processor(images=pil_img, return_tensors="pt")

        pixel_values = out.get("pixel_values", None)
        if pixel_values is None:
            raise RuntimeError(f"Processor output missing 'pixel_values'. Keys: {list(out.keys())}")

        grid_thw = out.get("image_grid_thw", None)
        if grid_thw is None:
            grid_thw = out.get("grid_thw", None)

        # --- 兼容 fast processor 可能直接返回 2D ---
        # Qwen2.5-VL patchified:
        #   pixel_values: (num_patches, cps) 或 (1, num_patches, cps)
        # Generic:
        #   pixel_values: (1, 3, H, W)
        if pixel_values.dim() == 2:
            # 已经是 (num_patches, cps) ✅
            pass
        elif pixel_values.dim() == 3:
            # (1, num_patches, cps) -> (num_patches, cps)
            pixel_values = pixel_values.squeeze(0)
            if pixel_values.dim() != 2:
                raise RuntimeError(f"After squeeze, unexpected pixel_values: {tuple(pixel_values.shape)}")
        elif pixel_values.dim() == 4:
            # (1,3,H,W) -> (3,H,W)（仅作兼容；Qwen2.5-VL 正常不会走这条）
            pixel_values = pixel_values.squeeze(0)
        else:
            raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

        # 如果是 patchified (2D)，则必须要 grid_thw
        if pixel_values.dim() == 2:
            if grid_thw is None:
                raise RuntimeError(
                    f"Patchified pixel_values requires grid_thw, but got None. Keys: {list(out.keys())}"
                )

            # grid_thw 可能是 (1,3) / (3,) / (1,1,3)
            if isinstance(grid_thw, torch.Tensor):
                if grid_thw.dim() == 3:
                    grid_thw = grid_thw.squeeze(0)
                if grid_thw.dim() == 1:
                    grid_thw = grid_thw.unsqueeze(0)
                if not (grid_thw.dim() == 2 and grid_thw.shape[-1] == 3):
                    raise RuntimeError(f"Unexpected grid_thw shape: {tuple(grid_thw.shape)}")
            else:
                raise RuntimeError(f"grid_thw is not a Tensor: {type(grid_thw)}")

        return pixel_values, grid_thw



    def __getitem__(self, idx):
        row = self.rows[idx]

        a_pv, a_grid = self._encode_vision_inputs(self._load_pil(row["anchor"]))
        p_pv, p_grid = self._encode_vision_inputs(self._load_pil(row["positive"]))
        n_pv, n_grid = self._encode_vision_inputs(self._load_pil(row["negative"]))

        delta = float(row.get("label", {}).get("negative_delta", 0.0))

        return {
            "anchor_pixel_values": a_pv,
            "anchor_grid_thw": a_grid,
            "positive_pixel_values": p_pv,
            "positive_grid_thw": p_grid,
            "negative_pixel_values": n_pv,
            "negative_grid_thw": n_grid,
            "negative_delta": torch.tensor(delta, dtype=torch.float),
        }


def qwen_vl_triplet_collator(features: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    - 若是 Qwen patchified：pixel_values 为 (num_patches, cps)，grid_thw 为 (1,3)；batch 需要在 patches 维拼起来
    - 若是普通 vision：pixel_values 为 (3,H,W)，grid_thw=None；batch 直接 stack 成 (B,3,H,W)
    """

    def is_patchified(x: torch.Tensor) -> bool:
        return x.dim() == 2  # (num_patches, cps)

    # anchor 的 pixel_values 判断模式即可
    patch_mode = is_patchified(features[0]["anchor_pixel_values"])

    def collate_side(prefix: str):
        pvs = [f[f"{prefix}_pixel_values"] for f in features]
        grids = [f.get(f"{prefix}_grid_thw", None) for f in features]

        if patch_mode:
            # concat patches: (sum_np, cps)
            pv = torch.cat(pvs, dim=0)
            # concat grids: (B, 3)
            if any(g is None for g in grids):
                raise RuntimeError("Patchified mode requires grid_thw, but got None.")
            grid = torch.cat(grids, dim=0)
            return pv, grid
        else:
            # stack images: (B, 3, H, W)
            pv = torch.stack(pvs, dim=0)
            return pv, None

    a_pv, a_grid = collate_side("anchor")
    p_pv, p_grid = collate_side("positive")
    n_pv, n_grid = collate_side("negative")

    deltas = torch.stack([f["negative_delta"] for f in features], dim=0)

    batch = {
        "anchor_pixel_values": a_pv,
        "positive_pixel_values": p_pv,
        "negative_pixel_values": n_pv,
        "negative_delta": deltas,
    }
    if patch_mode:
        batch.update({
            "anchor_grid_thw": a_grid,
            "positive_grid_thw": p_grid,
            "negative_grid_thw": n_grid,
        })
    return batch


# class VisionEmbeddingModel(nn.Module):
#     """
#     支持两类输入：
#     1) Qwen2.5-VL patchified: pixel_values=(sum_patches,cps), grid_thw=(num_images,3)
#     2) 普通 vision: pixel_values=(B,3,H,W), grid_thw=None
#     """
#     def __init__(self, model_name: str, gradient_checkpointing: bool = False):
#         super().__init__()
#         print(f"Loading model from: {model_name}")

#         self.encoder = None
#         self.is_qwen_vl = False

#         # --- 优先尝试 Qwen2.5-VL 专用类 ---
#         Qwen2_5_VLForConditionalGeneration = None
#         try:
#             from transformers import Qwen2_5_VLForConditionalGeneration as _Q
#             Qwen2_5_VLForConditionalGeneration = _Q
#         except Exception:
#             Qwen2_5_VLForConditionalGeneration = None

#         if Qwen2_5_VLForConditionalGeneration is not None:
#             try:
#                 print("Attempting to load via Qwen2_5_VLForConditionalGeneration...")
#                 full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#                     model_name,
#                     trust_remote_code=True,
#                     dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
#                     device_map="cpu",
#                 )
#                 if hasattr(full_model, "visual"):
#                     self.encoder = full_model.visual
#                     self.is_qwen_vl = True
#                     print("Successfully extracted 'visual' from Qwen2_5_VLForConditionalGeneration.")
#                 del full_model
#             except Exception as e:
#                 print(f"Qwen2_5_VL specific load failed: {e}")
#                 self.encoder = None
#                 self.is_qwen_vl = False

#         # --- 回退：AutoModel（比如纯 ViT）---
#         if self.encoder is None:
#             print("Falling back to AutoModel (generic)...")
#             try:
#                 full_model = AutoModel.from_pretrained(
#                     model_name,
#                     trust_remote_code=True,
#                     dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
#                     device_map="cpu",
#                 )
#                 if hasattr(full_model, "visual"):
#                     self.encoder = full_model.visual
#                 elif hasattr(full_model, "vision_tower"):
#                     self.encoder = full_model.vision_tower
#                 elif hasattr(full_model, "model") and hasattr(full_model.model, "visual"):
#                     self.encoder = full_model.model.visual
#                 else:
#                     self.encoder = full_model
#                 del full_model
#             except Exception as e:
#                 print(f"CRITICAL: Failed to load model: {e}", file=sys.stderr)
#                 sys.exit(1)

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         if gradient_checkpointing:
#             if hasattr(self.encoder, "gradient_checkpointing_enable"):
#                 self.encoder.gradient_checkpointing_enable()
#             elif hasattr(self.encoder, "enable_gradient_checkpointing"):
#                 self.encoder.enable_gradient_checkpointing()

#         print(f"Vision Tower ready. is_qwen_vl={self.is_qwen_vl}")

#     def _get_spatial_merge_unit(self) -> int:
#         # Qwen 系列通常有 spatial_merge_unit 或 spatial_merge_size（size^2）
#         unit = getattr(self.encoder, "spatial_merge_unit", None)
#         if unit is not None:
#             return int(unit)

#         size = getattr(self.encoder, "spatial_merge_size", None)
#         if size is None and hasattr(self.encoder, "config"):
#             size = getattr(self.encoder.config, "spatial_merge_size", None)
#         if size is not None:
#             size = int(size)
#             return size * size

#         return 1

#     def _pool_qwen_tokens(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
#         """
#         x: (sum_tokens, dim)
#         grid_thw: (num_images, 3) => [t, h, w]
#         token_count_per_image = (t*h*w) // spatial_merge_unit
#         """
#         if x.dim() == 3:
#             # 偶尔某些实现会直接返回 (B, seq, dim)
#             feats = x.mean(dim=1)
#             return F.normalize(feats, dim=-1)

#         if x.dim() != 2:
#             raise RuntimeError(f"Unexpected qwen visual output dim: {x.dim()} shape={tuple(x.shape)}")

#         merge_unit = self._get_spatial_merge_unit()
#         counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // merge_unit
#         counts = counts.to(torch.long).tolist()

#         feats = []
#         offset = 0
#         for c in counts:
#             c = int(c)
#             if c <= 0:
#                 raise RuntimeError(f"Non-positive token count computed from grid_thw: {counts}")
#             feats.append(x[offset:offset + c].mean(dim=0))
#             offset += c

#         if offset != x.shape[0]:
#             # 不强制 assert，给个提示（通常意味着 grid_thw 或 merge_unit 不匹配）
#             # 但仍然返回已切分的部分，避免直接 crash
#             print(f"[Warn] token slicing mismatch: used {offset} tokens, but x has {x.shape[0]} tokens.", file=sys.stderr)

#         feats = torch.stack(feats, dim=0)
#         return F.normalize(feats, dim=-1)

#     def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
#         """
#         - Qwen patchified: pixel_values (sum_patches, cps) + grid_thw (num_images,3)
#         - Generic: pixel_values (B,3,H,W)
#         """
#         # Qwen patchified
#         if pixel_values.dim() == 2:
#             if grid_thw is None:
#                 raise RuntimeError("pixel_values is patchified (2D) but grid_thw is None.")

#             try:
#                 outputs = self.encoder(pixel_values, grid_thw=grid_thw)
#             except TypeError:
#                 # 某些版本参数名可能不同/不需要 grid_thw
#                 outputs = self.encoder(pixel_values)

#             if isinstance(outputs, tuple):
#                 x = outputs[0]
#             elif hasattr(outputs, "last_hidden_state"):
#                 x = outputs.last_hidden_state
#             else:
#                 x = outputs

#             return self._pool_qwen_tokens(x, grid_thw)

#         # Generic vision encoder (B,3,H,W)
#         if pixel_values.dim() == 4:
#             try:
#                 outputs = self.encoder(pixel_values=pixel_values)
#             except TypeError:
#                 outputs = self.encoder(pixel_values)

#             if isinstance(outputs, tuple):
#                 x = outputs[0]
#             elif hasattr(outputs, "last_hidden_state"):
#                 x = outputs.last_hidden_state
#             else:
#                 x = outputs

#             # x: (B, seq, dim) or (B, dim)
#             if x.dim() == 3:
#                 feats = x.mean(dim=1)
#             elif x.dim() == 2:
#                 feats = x
#             else:
#                 raise RuntimeError(f"Unexpected generic visual output shape: {tuple(x.shape)}")

#             return F.normalize(feats, dim=-1)

#         raise RuntimeError(f"Unsupported pixel_values shape: {tuple(pixel_values.shape)}")


import gc
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

class VisionEmbeddingModel(nn.Module):
    """
    自动根据 model_path/config.json 判断是 qwen3_vl / qwen2_5_vl，然后用正确的 *ForConditionalGeneration
    来加载并抽取视觉塔（visual），避免 “newly initialized”。
    
    支持两类输入：
    1) Qwen VL patchified: pixel_values=(sum_patches,cps), grid_thw=(num_images,3)
    2) 普通 vision:       pixel_values=(B,3,H,W),         grid_thw=None
    """
    def __init__(
        self,
        model_name_or_path: str,
        gradient_checkpointing: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = "cpu",   # 你也可以传 None，让 Trainer/Accelerate 自己搬到 GPU
    ):
        super().__init__()
        print(f"Loading model from: {model_name_or_path}")

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        self.encoder = None
        self.is_qwen_vl = False

        # 1) 读 config 来判断模型类型
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_type = getattr(config, "model_type", "") or ""
            archs = getattr(config, "architectures", None) or []
            model_type_l = str(model_type).lower()
            archs_l = " ".join([str(a).lower() for a in archs])
            sig = (model_type_l + " " + archs_l).strip()
            print(f"[Detect] model_type={model_type} architectures={archs}")
        except Exception as e:
            print(f"[Warn] AutoConfig load failed, fallback detection by name only: {e}", file=sys.stderr)
            sig = model_name_or_path.lower()

        # 2) 根据签名选择优先加载的专用类（从 transformers import）
        candidate_cls_names = []
        if "qwen3_vl" in sig or "qwen3vl" in sig:
            # Qwen3-VL
            candidate_cls_names = [
                "Qwen3VLForConditionalGeneration",
            ]
            self.is_qwen_vl = True
        elif "qwen2_5_vl" in sig or "qwen2.5-vl" in sig or "qwen2_5vl" in sig:
            # Qwen2.5-VL
            candidate_cls_names = [
                "Qwen2_5_VLForConditionalGeneration",
            ]
            self.is_qwen_vl = True
        elif "qwen2_vl" in sig or "qwen2vl" in sig:
            # 某些环境可能有 Qwen2VLForConditionalGeneration
            candidate_cls_names = [
                "Qwen2VLForConditionalGeneration",
            ]
            self.is_qwen_vl = True

        def _extract_visual(full_model: nn.Module) -> nn.Module:
            # 尽量覆盖常见命名
            if hasattr(full_model, "visual"):
                return full_model.visual
            if hasattr(full_model, "vision_tower"):
                return full_model.vision_tower
            if hasattr(full_model, "model") and hasattr(full_model.model, "visual"):
                return full_model.model.visual
            if hasattr(full_model, "model") and hasattr(full_model.model, "vision_tower"):
                return full_model.model.vision_tower
            # 实在不行就返回整个模型（不推荐，但能跑）
            return full_model

        # 3) 优先走专用类加载（关键：避免权重对不上导致 newly initialized）
        last_err = None
        if candidate_cls_names:
            for cls_name in candidate_cls_names:
                try:
                    cls = getattr(__import__("transformers", fromlist=[cls_name]), cls_name)
                except Exception as e:
                    last_err = e
                    continue

                try:
                    print(f"Attempting to load via {cls_name}...")
                    full_model = cls.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                    )
                    self.encoder = _extract_visual(full_model)
                    # 释放语言塔等无关部分的引用（encoder 作为子模块仍会保留参数）
                    del full_model
                    print(f"Successfully extracted visual tower via {cls_name}.")
                    break
                except Exception as e:
                    last_err = e
                    print(f"{cls_name} load failed: {e}", file=sys.stderr)
                    self.encoder = None

        # 4) 仍失败才 fallback 到 AutoModel（注意：这一步可能再次出现权重不完整的风险）
        if self.encoder is None:
            print("Falling back to AutoModel (generic)...", file=sys.stderr)
            if last_err is not None:
                print(f"[Prev error] {last_err}", file=sys.stderr)
            try:
                full_model = AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                self.encoder = _extract_visual(full_model)
                del full_model
                # generic 情况下我们不敢保证是 qwen patchified
                self.is_qwen_vl = False
            except Exception as e:
                print(f"CRITICAL: Failed to load model: {e}", file=sys.stderr)
                sys.exit(1)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            elif hasattr(self.encoder, "enable_gradient_checkpointing"):
                self.encoder.enable_gradient_checkpointing()

        print(f"Vision Tower ready. is_qwen_vl={self.is_qwen_vl}")

    def _get_spatial_merge_unit(self) -> int:
        unit = getattr(self.encoder, "spatial_merge_unit", None)
        if unit is not None:
            return int(unit)

        size = getattr(self.encoder, "spatial_merge_size", None)
        if size is None and hasattr(self.encoder, "config"):
            size = getattr(self.encoder.config, "spatial_merge_size", None)
        if size is not None:
            size = int(size)
            return size * size
        return 1

    def _pool_qwen_tokens(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # x: (sum_tokens, dim) or (B, seq, dim)
        if x.dim() == 3:
            feats = x.mean(dim=1)
            return F.normalize(feats, dim=-1)

        if x.dim() != 2:
            raise RuntimeError(f"Unexpected qwen visual output dim: {x.dim()} shape={tuple(x.shape)}")

        merge_unit = self._get_spatial_merge_unit()
        counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // merge_unit
        counts = counts.to(torch.long).tolist()

        feats = []
        offset = 0
        for c in counts:
            c = int(c)
            if c <= 0:
                raise RuntimeError(f"Non-positive token count computed from grid_thw: {counts}")
            feats.append(x[offset:offset + c].mean(dim=0))
            offset += c

        if offset != x.shape[0]:
            print(f"[Warn] token slicing mismatch: used {offset} tokens, but x has {x.shape[0]} tokens.",
                  file=sys.stderr)

        feats = torch.stack(feats, dim=0)
        return F.normalize(feats, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # Qwen patchified: (sum_patches, cps)
        if pixel_values.dim() == 2:
            if grid_thw is None:
                raise RuntimeError("pixel_values is patchified (2D) but grid_thw is None.")
            try:
                outputs = self.encoder(pixel_values, grid_thw=grid_thw)
            except TypeError:
                # 某些实现参数名不同或不需要 grid_thw
                outputs = self.encoder(pixel_values)

            if isinstance(outputs, tuple):
                x = outputs[0]
            elif hasattr(outputs, "last_hidden_state"):
                x = outputs.last_hidden_state
            else:
                x = outputs

            return self._pool_qwen_tokens(x, grid_thw)

        # Generic vision: (B,3,H,W)
        if pixel_values.dim() == 4:
            try:
                outputs = self.encoder(pixel_values=pixel_values)
            except TypeError:
                outputs = self.encoder(pixel_values)

            if isinstance(outputs, tuple):
                x = outputs[0]
            elif hasattr(outputs, "last_hidden_state"):
                x = outputs.last_hidden_state
            else:
                x = outputs

            if x.dim() == 3:
                feats = x.mean(dim=1)
            elif x.dim() == 2:
                feats = x
            else:
                raise RuntimeError(f"Unexpected generic visual output shape: {tuple(x.shape)}")

            return F.normalize(feats, dim=-1)

        raise RuntimeError(f"Unsupported pixel_values shape: {tuple(pixel_values.shape)}")

class TripletTrainer(Trainer):
    def __init__(self, margin=0.3, delta_weight=False, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.delta_weight = delta_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        delta = inputs["negative_delta"]

        # Qwen patchified 分支：pixel_values 是 2D，且带 grid_thw
        if inputs["anchor_pixel_values"].dim() == 2:
            a_pv, a_grid = inputs["anchor_pixel_values"], inputs["anchor_grid_thw"]
            p_pv, p_grid = inputs["positive_pixel_values"], inputs["positive_grid_thw"]
            n_pv, n_grid = inputs["negative_pixel_values"], inputs["negative_grid_thw"]

            pv = torch.cat([a_pv, p_pv, n_pv], dim=0)
            grid = torch.cat([a_grid, p_grid, n_grid], dim=0)

            emb = model(pv, grid)  # (3B, dim)
        else:
            # Generic 分支：pixel_values 是 4D (B,3,H,W)
            anchor = inputs["anchor_pixel_values"]
            positive = inputs["positive_pixel_values"]
            negative = inputs["negative_pixel_values"]

            combined = torch.cat([anchor, positive, negative], dim=0)
            emb = model(combined)  # (3B, dim)

        a, p, n = torch.chunk(emb, 3, dim=0)

        sim_ap = torch.sum(a * p, dim=-1)
        sim_an = torch.sum(a * n, dim=-1)

        d_ap = 1.0 - sim_ap
        d_an = 1.0 - sim_an

        losses = F.relu(d_ap - d_an + self.margin)

        if self.delta_weight:
            weights = 1.0 + torch.log1p(delta.abs())
            losses = losses * weights

        loss = losses.mean()

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train_loss": float(loss.detach().cpu()),
                "cos_ap": float(sim_ap.mean().detach().cpu()),
                "cos_an": float(sim_an.mean().detach().cpu()),
            })

        return (loss, None) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", required=True)

    # Training Hyperparams
    parser.add_argument("--resolution", type=int, default=0, help="Optional override for processor.size if supported; 0 means use model default.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.3)

    # Options
    parser.add_argument("--delta_weight", action="store_true")
    parser.add_argument("--triplet_type", default=None)
    parser.add_argument("--resume_from", default=None)

    # Optimization Flags
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Better for Qwen/LLaVA training")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--workers", type=int, default=8)

    # Processor behavior
    parser.add_argument("--use_fast_processor", action="store_true", help="Force use_fast=True for image processor.")
    parser.add_argument("--use_slow_processor", action="store_true", help="Force use_fast=False for image processor.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fp16 and args.bf16:
        print("Warning: Both fp16 and bf16 passed. Preferring bf16.")
        args.fp16 = False

    if args.use_fast_processor and args.use_slow_processor:
        print("Warning: Both --use_fast_processor and --use_slow_processor set. Using fast.")
        args.use_slow_processor = False

    use_fast = None
    if args.use_fast_processor:
        use_fast = True
    elif args.use_slow_processor:
        use_fast = False

    dataset = TripletClockDataset(
        jsonl_path=args.jsonl,
        root_dir=args.root_dir,
        model_name=args.model_name,
        resolution=args.resolution,
        triplet_type_filter=args.triplet_type,
        use_fast_processor=use_fast,
    )

    model = VisionEmbeddingModel(
        args.model_name,
        gradient_checkpointing=args.grad_checkpoint,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    trainer = TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=qwen_vl_triplet_collator,
        margin=args.margin,
        delta_weight=args.delta_weight,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)

    # 保存 processor（用于推理复现）
    try:
        kwargs = {"trust_remote_code": True}
        if use_fast is not None:
            kwargs["use_fast"] = use_fast
        AutoImageProcessor.from_pretrained(args.model_name, **kwargs).save_pretrained(args.output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
