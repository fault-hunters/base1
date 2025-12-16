"""
두 이미지 경로와 가중치 경로를 입력받아 스타일/컨텐츠 OK·NG를 반환하는 API + CLI.
인자를 생략하면 실행 시 입력()으로 받아줍니다.
"""

import argparse
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
from torchvision import transforms
from sconf import Config
from models import Generator


def _build_transform(cfg):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3) if cfg.dset_aug.normalize else lambda x: x,
    ])


def _load_gen(cfg, weight_path: Path, device: torch.device) -> Generator:
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "gen" in state:
        state = state["gen"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    gen.load_state_dict(state)
    gen.eval()
    return gen


@torch.no_grad()
def compare_images(imgA_path: Path, imgB_path: Path, weight_path: Path, cfg_path: Path = None) -> Dict[str, float]:
    """
    imgA_path, imgB_path: 비교할 이미지 경로
    weight_path: 학습된 generator 가중치(.pth)
    cfg_path: sconf 설정 파일 경로 (기본: mxfont/cfgs/eval.yaml)
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

    thr = cfg.threshold
    return {
        "style_sim": float(sim_s.item()),
        "content_sim": float(sim_c.item()),
        "style_pred": "OK" if sim_s.item() >= thr else "NG",
        "content_pred": "OK" if sim_c.item() >= thr else "NG",
        "threshold": float(thr),
    }


def main():
    ap = argparse.ArgumentParser(description="두 이미지 스타일/컨텐츠 비교")
    ap.add_argument("--cfg", default=None, help="설정 파일 경로 (기본: mxfont/cfgs/eval.yaml)")
    ap.add_argument("--weight", required=False, help="generator 가중치 경로 (.pth)")
    ap.add_argument("--imgA", required=False, help="비교할 첫 번째 이미지 경로")
    ap.add_argument("--imgB", required=False, help="비교할 두 번째 이미지 경로")
    args = ap.parse_args()

    # 인자를 생략하면 실행 시 입력()으로 받음
    weight_path = Path(args.weight) if args.weight else Path(input("가중치 경로(.pth)를 입력하세요: ").strip())
    imgA_path = Path(args.imgA) if args.imgA else Path(input("첫 번째 이미지 경로를 입력하세요: ").strip())
    imgB_path = Path(args.imgB) if args.imgB else Path(input("두 번째 이미지 경로를 입력하세요: ").strip())

    result = compare_images(
        imgA_path=imgA_path,
        imgB_path=imgB_path,
        weight_path=weight_path,
        cfg_path=Path(args.cfg) if args.cfg else None,
    )

    print(
        f"style_sim={result['style_sim']:.3f} -> {result['style_pred']} "
        f"(thr={result['threshold']})"
    )
    print(
        f"content_sim={result['content_sim']:.3f} -> {result['content_pred']} "
        f"(thr={result['threshold']})"
    )


if __name__ == "__main__":
    main()
