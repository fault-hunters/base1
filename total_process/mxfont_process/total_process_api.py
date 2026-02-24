"""
두 이미지 경로와 가중치 경로를 입력받아 스타일/컨텐츠 OK·NG를 반환하는 API + CLI.
인자를 생략하면 실행 시 입력()으로 받아줍니다.
"""
import torch
import argparse
from pathlib import Path
from typing import Dict
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from sconf import Config
from mxfont_process.models.generator2 import Generator
from . import utils
from PIL import Image, ImageOps

class SquarePad:
    def __init__(self, fill=255):  # 흰색 배경: 255, 검정: 0, RGB면 (255,255,255)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        size = max(w, h)
        pad_left = (size - w) // 2
        pad_top = (size - h) // 2
        pad_right = size - w - pad_left
        pad_bottom = size - h - pad_top
        return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

def _build_transform(cfg):
    return transforms.Compose([
        SquarePad(fill=(255, 255, 255)),
        transforms.Resize((1024, 1024)), # input img resizing 1024X1024
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3) if cfg.dset_aug.normalize else lambda x: x,
    ])


def _load_gen(cfg, weight_path: Path, device: torch.device) -> Generator:
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    state = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "gen" in state:
        state = state["gen"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    gen.load_state_dict(state)
    gen.eval()
    return gen

@torch.no_grad()
def compare_images(
    imgA_path: Path,
    imgB_path: Path,
    weight_path: Path,
    cfg_path: Path = None
) -> Dict[str, float]:
    """
    imgA_path, imgB_path: 비교할 이미지 경로
    weight_path: 학습된 generator 가중치(.pth)
    cfg_path: sconf 설정 파일 경로(선택). 없으면 defaults.yaml만 사용
    """
    base_dir = Path(__file__).parent
    cfg_paths = [cfg_path] if cfg_path else []
    cfg = Config(*cfg_paths, default=base_dir / "cfgs" / "defaults.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = _build_transform(cfg)

    def _load_img(p):
        return transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

    imgA = _load_img(imgA_path)
    imgB = _load_img(imgB_path)

    gen = _load_gen(cfg, weight_path, device)
    sA, cA = gen.extract_style_content(imgA)
    sB, cB = gen.extract_style_content(imgB)

    sim_s = torch.nn.functional.cosine_similarity(sA, sB, dim=1).clamp(-1, 1)
    sim_c = torch.nn.functional.cosine_similarity(cA, cB, dim=1).clamp(-1, 1)
    sim_s = (sim_s + 1) / 2 # 0~1범위로 스케일링
    sim_c = (sim_c + 1) / 2 # 0~1범위로 스케일링

    thr_s = cfg.threshold_s
    thr_c = cfg.threshold_c

    return {
        "style_sim": float(sim_s.item()),
        "content_sim": float(sim_c.item()),
        "style_pred": "Font OK" if sim_s.item() >= thr_s else "Font NG",
        "content_pred": "Letter OK" if sim_c.item() >= thr_c else "Letter NG",
        "threshold_c": float(thr_c),
        "threshold_s": float(thr_s)
    }

# pair 하나당 탐지
def infer_from_path(img_path_a: str, img_path_b: str, weight_path: str, cfg_path: str) -> str:
    result = compare_images(
        imgA_path=img_path_a,
        imgB_path=img_path_b,
        weight_path=weight_path,
        cfg_path=cfg_path
    )

    print(
        f"style_sim={result['style_sim']:.3f} -> {result['style_pred']} "
        f"(thr(s)={result['threshold_s']})"
    )
    print(
        f"content_sim={result['content_sim']:.3f} -> {result['content_pred']} "
        f"(thr(c)={result['threshold_c']})"
    )

    return result


if __name__ == "__main__":
    infer_from_path()
