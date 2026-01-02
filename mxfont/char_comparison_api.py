"""
두 이미지 경로와 가중치 경로를 입력받아 스타일/컨텐츠 OK·NG를 반환하는 API + CLI.
인자를 생략하면 실행 시 입력()으로 받아줍니다.
"""

import argparse
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from sconf import Config
from models import Generator
import utils


def _build_transform(cfg):
    return transforms.Compose([
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
    gen.load_state_dict(state)
    gen.eval()
    return gen


@torch.no_grad()
def _extract_style_content_maps(gen: Generator, img: torch.Tensor):
    feats = gen.encode(img)
    style_facts = gen.factorize(feats, 0)
    char_facts = gen.factorize(feats, 1)

    style_last = style_facts["last"].mean(1)
    style_skip = style_facts["skip"].mean(1)
    char_last = char_facts["last"].mean(1)
    char_skip = char_facts["skip"].mean(1)

    style_feat = gen.fuser_style(style_last, style_skip)
    content_feat = gen.fuser_content(char_last, char_skip)
    return style_feat, content_feat


def _cosine_sim_map(featA: torch.Tensor, featB: torch.Tensor, out_hw=None) -> torch.Tensor:
    sim = F.cosine_similarity(featA, featB, dim=1).clamp(-1, 1)  # (B,H,W)
    sim = (sim + 1) / 2  # 0~1
    if out_hw is not None:
        sim = F.interpolate(sim.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False).squeeze(1)
    return sim


@torch.no_grad()
def compare_images(
    imgA_path: Path,
    imgB_path: Path,
    weight_path: Path,
    cfg_path: Path = None,
    save_dir: Path = None,
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
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if cfg.dset_aug.normalize:
            imgA_vis = (imgA * 0.5 + 0.5).clamp(0, 1)
            imgB_vis = (imgB * 0.5 + 0.5).clamp(0, 1)
        else:
            imgA_vis = imgA.clamp(0, 1)
            imgB_vis = imgB.clamp(0, 1)

        utils.save_tensor_to_image(imgA_vis[0].cpu(), save_dir / "imgA.png")
        utils.save_tensor_to_image(imgB_vis[0].cpu(), save_dir / "imgB.png")

        styleA, contentA = _extract_style_content_maps(gen, imgA)
        styleB, contentB = _extract_style_content_maps(gen, imgB)
        out_hw = imgA.shape[2:]
        style_map = _cosine_sim_map(styleA, styleB, out_hw=out_hw)
        content_map = _cosine_sim_map(contentA, contentB, out_hw=out_hw)
        utils.save_tensor_to_image(style_map[0].unsqueeze(0).cpu(), save_dir / "style_sim_map.png")
        utils.save_tensor_to_image(content_map[0].unsqueeze(0).cpu(), save_dir / "content_sim_map.png")

    return {
        "style_sim": float(sim_s.item()),
        "content_sim": float(sim_c.item()),
        "style_pred": "OK" if sim_s.item() >= thr_s else "NG",
        "content_pred": "OK" if sim_c.item() >= thr_c else "NG",
        "threshold_c": float(thr_c),
        "threshold_s": float(thr_s)
    }


def main():
    ap = argparse.ArgumentParser(description="두 이미지 스타일/컨텐츠 비교")
    ap.add_argument("--cfg", default=None, help="설정 파일 경로(선택). 없으면 defaults.yaml만 사용")
    ap.add_argument("--weight", required=False, help="generator 가중치 경로 (.pth)")
    ap.add_argument("--imgA", required=False, help="비교할 첫 번째 이미지 경로")
    ap.add_argument("--imgB", required=False, help="비교할 두 번째 이미지 경로")
    ap.add_argument("--save_dir", default='./test_map', help="지정 시 imgA/imgB + 유사도 히트맵 PNG 저장 디렉터리")
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
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )

    print(
        f"style_sim={result['style_sim']:.3f} -> {result['style_pred']} "
        f"(thr(s)={result['threshold_s']})"
    )
    print(
        f"content_sim={result['content_sim']:.3f} -> {result['content_pred']} "
        f"(thr(c)={result['threshold_c']})"
    )


if __name__ == "__main__":
    main()
