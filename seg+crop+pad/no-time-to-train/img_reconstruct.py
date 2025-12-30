import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import process_single_pair

# segmentation
# =====================================================
# 모델 초기화
# =====================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # DINOv2 모델 로드
    print("Loading DINOv2 model...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    dinov2_model.eval()
    print("✅ DINOv2 loaded")

    # SAM2 모델 로드
    print("Loading SAM2 model...")
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    print("✅ SAM2 loaded")

    # =====================================================
    # 메인 처리
    # =====================================================
    data = pd.read_csv("./data.csv",sep=',', encoding='utf-8')

    save_dir = "./seg_result"

    os.makedirs(save_dir, exist_ok=True)
    for _, pair in data.iterrows():
        ref_image_path = pair['ref_path']
        target_image_path = pair['tar_path']
        if not os.path.exists(ref_image_path):
            print(f"❌ Error: Ref image not found at {ref_image_path}")
        elif not os.path.exists(target_image_path):
            print(f"❌ Error: Target image not found at {target_image_path}")
        else:
            print("✅ Both images found. Starting processing...\n")
            print("="*70)

            try:
                result = process_single_pair(
                    ref_image_path,
                    target_image_path,
                    save_dir,
                    dinov2_model,
                    mask_generator,
                    device
                )

                print("\n" + "="*70)
                print("✅ Processing completed!")
                print("="*70)

                # 결과 출력
                print(f"\n📊 Results:")
                print(f"  - Output File:       {result['output_filename']}")
                print(f"  - Combined Sim:      {result['similarity']:.4f}")
                print(f"  - CLS Similarity:    {result['cls_similarity']:.4f}")
                print(f"  - Patch Similarity:  {result['patch_similarity']:.4f}")

                # CSV 저장
                df = pd.DataFrame([result])
                csv_path = os.path.join(save_dir, "similarity_scores.csv")
                df.to_csv(csv_path, index=False)

                print(f"\n💾 Results saved:")
                print(f"  - Segmented Image: {os.path.join(save_dir, result['output_filename'])}")
                print(f"  - CSV File:        {csv_path}")

            except Exception as e:
                print(f"\n❌ Error during processing: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()