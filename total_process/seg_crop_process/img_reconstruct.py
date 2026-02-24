import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import numpy as np
import cv2
import torch.nn.functional as F
import os
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
from .sam2.build_sam import build_sam2
from .sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .utils_seg import process_single_pair

# segmentation
# =====================================================
# 모델 초기화
# =====================================================
def seg_crop(ref_image_path, target_image_path, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DINOv2 모델 로드
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    dinov2_model.eval()

    # SAM2 모델 로드
    BASE = Path(__file__).resolve().parent.parent  # total_process/seg_crop_process -> total_process
    sam2_checkpoint = str(BASE / "checkpoints" / "sam2_hiera_large.pt")
    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        points_per_batch=32,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # =====================================================
    # 메인 처리
    # =====================================================
    result = process_single_pair(
        ref_image_path,
        target_image_path,
        save_dir,
        dinov2_model,
        mask_generator,
        device
    )

    # 결과 출력
    print(f"\n📊 Results:")
    print(f"  - Output File:       {result['output_filename']}")
    print(f"  - Combined Sim:      {result['similarity']:.4f}")
    print(f"  - CLS Similarity:    {result['cls_similarity']:.4f}")
    print(f"  - Patch Similarity:  {result['patch_similarity']:.4f}")

    return result

if __name__ == "__main__":
    seg_crop()