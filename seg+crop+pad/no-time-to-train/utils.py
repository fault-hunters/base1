import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import math


def load_image(image_path):
    """이미지 로드 및 RGB 변환"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def extract_dinov2_patch_features(image, model, device):
    """DINOv2 patch-level 특징 추출"""
    # 이미지 전처리
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # ImageNet 정규화
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # 518x518로 리사이즈 (37x37 패치)
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(518, 518),
        mode='bilinear',
        align_corners=False
    ).to(device)

    # Patch 특징 추출
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_features = features['x_norm_patchtokens'].squeeze(0)
        cls_feature = features['x_norm_clstoken'].squeeze(0)

    return patch_features, cls_feature

def compute_patch_similarity(ref_patches, ref_cls, target_patches, target_cls, target_mask=None):
    """
    Patch-level + CLS token 조합 유사도 계산
    target_mask: 37x37 크기의 마스크 (해당 패치만 비교)
    """
    # 1. CLS token 유사도 (전역 특징)
    ref_cls_norm = F.normalize(ref_cls.unsqueeze(0), p=2, dim=1)
    target_cls_norm = F.normalize(target_cls.unsqueeze(0), p=2, dim=1)
    cls_sim = (ref_cls_norm * target_cls_norm).sum().item()

    # 2. Patch-level 유사도
    if target_mask is not None:
        # 마스크 영역의 패치만 선택
        mask_flat = target_mask.flatten()
        valid_indices = mask_flat > 0

        if valid_indices.sum() == 0:
            patch_sim = 0.0
        else:
            target_patches_masked = target_patches[valid_indices]

            # Ref의 모든 패치와 Target의 마스크 패치 간 평균 유사도
            ref_norm = F.normalize(ref_patches, p=2, dim=1)
            target_norm = F.normalize(target_patches_masked, p=2, dim=1)

            # 각 target 패치와 가장 유사한 ref 패치 찾기
            similarity_matrix = torch.mm(target_norm, ref_norm.t())
            max_sims = similarity_matrix.max(dim=1)[0]
            patch_sim = max_sims.mean().item()
    else:
        # 마스크 없으면 전체 패치 평균
        ref_norm = F.normalize(ref_patches, p=2, dim=1)
        target_norm = F.normalize(target_patches, p=2, dim=1)
        patch_sim = (ref_norm * target_norm).sum(dim=1).mean().item()

    # 3. 조합 (CLS 30% + Patch 70%)
    combined_sim = 0.3 * cls_sim + 0.7 * patch_sim

    return combined_sim, cls_sim, patch_sim

def create_white_background_image(image, mask):
    """
    흰색 배경에 segmentation된 객체만 표시

    Args:
        image: 원본 이미지 (H, W, 3)
        mask: boolean 마스크 (H, W)

    Returns:
        흰색 배경 이미지 (H, W, 3)
    """
    h, w = image.shape[:2]

    # 흰색 배경 생성 (255, 255, 255)
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 마스크 영역만 원본 이미지로 채우기
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    result = np.where(mask_3ch, image, white_bg)

    return result.astype(np.uint8)

# bounding box
def get_bounding_box(mask):
    """마스크의 bounding box 좌표 계산"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def crop_by_mask(image, mask, padding=10):
    """마스크 영역을 bounding box로 크롭 (패딩 추가)"""
    bbox = get_bounding_box(mask)
    if bbox is None:
        return None

    rmin, rmax, cmin, cmax = bbox
    h, w = image.shape[:2]

    # 패딩 추가 (경계 넘지 않게)
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding)

    # 크롭
    cropped = image[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]

    return cropped, cropped_mask, rmin, rmax, cmin, cmax

def final_img_padding(img, pad_tb, pad_lr):
    # pad: int 하나(사방 동일) 또는 (top, bottom, left, right) 튜플
    top = bottom = pad_tb
    left = right = pad_lr

    if img.ndim == 2:  # 흑백
        pad_width = ((top, bottom), (left, right))
        value = 255
    else:              # RGB
        pad_width = ((top, bottom), (left, right), (0, 0))
        value = (255, 255, 255)

    return np.pad(img, pad_width, mode="constant", constant_values=value).astype(np.uint8)

def process_single_pair(ref_path, target_path, save_folder, dinov2_model, mask_generator, device):
    """
    단일 ref-target 쌍을 처리하는 함수
    """
    # 이미지 로딩
    ref_img = load_image(ref_path)
    target_img = load_image(target_path)

    # Ref 이미지의 DINOv2 특징 추출
    ref_patches, ref_cls = extract_dinov2_patch_features(ref_img, dinov2_model, device)

    # SAM2로 target 이미지의 모든 mask 생성
    target_masks_raw = mask_generator.generate(target_img)

    print(f"  Generated {len(target_masks_raw)} masks")

    # 각 마스크에 대해 유사도 계산
    mask_similarities = []

    for idx, mask_data in enumerate(target_masks_raw):
        mask = mask_data['segmentation']

        # 마스크 크기 필터링
        mask_area = mask.sum() / (mask.shape[0] * mask.shape[1])

        if 0.02 < mask_area < 0.85: # 유효한 mask
            # Bounding box로 크롭
            crop_result = crop_by_mask(target_img, mask, padding=10)

            if crop_result is not None:
                cropped_img, cropped_mask, bbox_rmin, bbox_rmax, bbox_cmin, bbox_cmax = crop_result

                # 크롭된 이미지의 특징 추출
                crop_patches, crop_cls = extract_dinov2_patch_features(cropped_img, dinov2_model)

                # 크롭된 마스크를 37x37로 리사이즈
                cropped_mask_resized = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (37, 37),
                    interpolation=cv2.INTER_NEAREST
                )

                # Patch-level 유사도 계산
                combined_sim, cls_sim, patch_sim = compute_patch_similarity(
                    ref_patches, ref_cls,
                    crop_patches, crop_cls,
                    cropped_mask_resized
                )

                mask_similarities.append({
                    'idx': idx,
                    'similarity': combined_sim,
                    'cls_sim': cls_sim,
                    'patch_sim': patch_sim,
                    'mask': mask,
                    'area': mask_area,
                    'rmin' : bbox_rmin,
                    'rmax' : bbox_rmax,
                    'cmin' : bbox_cmin,
                    'cmax' : bbox_cmax
                })
            else:
                mask_similarities.append({
                    'idx': idx,
                    'similarity': -1.0,
                    'cls_sim': -1.0,
                    'patch_sim': -1.0,
                    'mask': mask,
                    'area': mask_area,
                    'rmin' : bbox_rmin,
                    'rmax' : bbox_rmax,
                    'cmin' : bbox_cmin,
                    'cmax' : bbox_cmax
                })
        else:
            mask_similarities.append({
                'idx': idx,
                'similarity': -1.0,
                'cls_sim': -1.0,
                'patch_sim': -1.0,
                'mask': mask,
                'area': mask_area,
                'rmin' : bbox_rmin,
                'rmax' : bbox_rmax,
                'cmin' : bbox_cmin,
                'cmax' : bbox_cmax
            })

    # 유사도가 가장 높은 마스크 선택
    valid_masks = [m for m in mask_similarities if m['similarity'] > 0]

    if len(valid_masks) == 0:
        # 유효한 마스크가 없으면 첫 번째 마스크 사용
        best_mask_info = mask_similarities[0]
        print(f"  ⚠️ No valid masks, using first mask")
    else:
        best_mask_info = max(valid_masks, key=lambda x: x['similarity'])
        print(f"  ✅ Best mask: #{best_mask_info['idx']} | " +
              f"Combined: {best_mask_info['similarity']:.4f} | " +
              f"CLS: {best_mask_info['cls_sim']:.4f} | " +
              f"Patch: {best_mask_info['patch_sim']:.4f} | " +
              f"Area: {best_mask_info['area']:.2%}")

    # 최적 마스크로 흰색 배경 이미지 생성
    final_mask = best_mask_info['mask']
    white_bg_image = create_white_background_image(target_img, final_mask)

    # cropping
    rmin = best_mask_info["rmin"]
    rmax = best_mask_info["rmax"]
    cmin = best_mask_info["cmin"]
    cmax = best_mask_info["cmax"]
    white_bg_image = white_bg_image[rmin:rmax+1, cmin:cmax+1]  # bbox 범위로 잘라 저장
    r_length = rmax-rmin # top, bottom
    c_length = cmax-cmin # left, right
    diag = math.sqrt(r_length ** 2 + c_length ** 2)
    diag = math.ceil(diag)+10
    r_pad = diag-r_length # top, bottom
    c_pad = diag-c_length # left, right
    white_bg_image_pad = final_img_padding(white_bg_image, r_pad, c_pad)

    # 파일명 생성 및 저장
    base_name = os.path.splitext(os.path.basename(target_path))[0]
    save_name = f"{base_name}@seg.png"
    save_path = os.path.join(save_folder, save_name)
    Image.fromarray(white_bg_image_pad).save(save_path)

    # 결과 반환
    return {
        'output_filename': save_name,
        'similarity': best_mask_info['similarity'],
        'cls_similarity': best_mask_info['cls_sim'],
        'patch_similarity': best_mask_info['patch_sim']
    }