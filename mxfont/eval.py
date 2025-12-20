import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from sconf import Config
# from models import Generator
from models.generator2 import Generator
from datasets_img import get_img_loader
import utils

def build_transform(cfg):
    ts = []
    rotation_deg = getattr(cfg.dset_aug, "rotation_deg", None)
    rotation_p = getattr(cfg.dset_aug, "rotation_p", 0.0)
    if rotation_deg is not None and rotation_p and rotation_p > 0:
        ts.append(transforms.RandomApply([transforms.RandomRotation(rotation_deg, fill=0)], p=rotation_p))
    ts.extend([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3) if cfg.dset_aug.normalize else lambda x: x,
    ])
    return transforms.Compose(ts)


@torch.no_grad()
def _extract_style_content_maps(gen: Generator, img: torch.Tensor):
    style_facts = gen.style_encode(img)
    char_facts = gen.content_encode(img)
    style_facts = gen.factorize_s(style_facts, 0)
    char_facts = gen.factorize_c(char_facts, 1)

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
def evaluate(gen, loader, device, threshold_s, threshold_c, vis_dir: Path = None, vis_n: int = 0, normalize: bool = True):
    gen.eval()
    total_s = total_c = total = 0
    tp_s = fp_s = fn_s = tn_s = 0
    tp_c = fp_c = fn_c = tn_c = 0
    saved = 0
    for imgA, imgB, label_s, label_c in loader:
        imgA = imgA.to(device); imgB = imgB.to(device)
        label_s = label_s.to(device).view(-1)
        label_c = label_c.to(device).view(-1)
        bs = label_s.size(0)
        sA, cA = gen.extract_style_content(imgA)
        sB, cB = gen.extract_style_content(imgB)
        sim_s = torch.nn.functional.cosine_similarity(sA, sB, dim=1).clamp(-1, 1)
        sim_c = torch.nn.functional.cosine_similarity(cA, cB, dim=1).clamp(-1, 1)
        sim_s = (sim_s + 1) / 2
        sim_c = (sim_c + 1) / 2

        pred_s = (sim_s >= threshold_s).float()
        pred_c = (sim_c >= threshold_c).float()
        acc_s = (pred_s == label_s).float().mean()
        acc_c = (pred_c == label_c).float().mean()
        total_s += acc_s * bs
        total_c += acc_c * bs
        total += bs
        tp_s += ((pred_s == 1) & (label_s == 1)).sum().item()
        fp_s += ((pred_s == 1) & (label_s == 0)).sum().item()
        fn_s += ((pred_s == 0) & (label_s == 1)).sum().item()
        tn_s += ((pred_s == 0) & (label_s == 0)).sum().item()
        tp_c += ((pred_c == 1) & (label_c == 1)).sum().item()
        fp_c += ((pred_c == 1) & (label_c == 0)).sum().item()
        fn_c += ((pred_c == 0) & (label_c == 1)).sum().item()
        tn_c += ((pred_c == 0) & (label_c == 0)).sum().item()

        if vis_dir is not None and vis_n and saved < vis_n:
            vis_dir = Path(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)

            styleA, contentA = _extract_style_content_maps(gen, imgA)
            styleB, contentB = _extract_style_content_maps(gen, imgB)
            out_hw = imgA.shape[2:]
            style_map = _cosine_sim_map(styleA, styleB, out_hw=out_hw)
            content_map = _cosine_sim_map(contentA, contentB, out_hw=out_hw)

            if normalize:
                imgA_vis = (imgA * 0.5 + 0.5).clamp(0, 1)
                imgB_vis = (imgB * 0.5 + 0.5).clamp(0, 1)
            else:
                imgA_vis = imgA.clamp(0, 1)
                imgB_vis = imgB.clamp(0, 1)

            for bi in range(bs):
                if saved >= vis_n:
                    break
                prefix = f"{saved:06d}"
                utils.save_tensor_to_image(imgA_vis[bi].cpu(), vis_dir / f"{prefix}_imgA.png")
                utils.save_tensor_to_image(imgB_vis[bi].cpu(), vis_dir / f"{prefix}_imgB.png")
                utils.save_tensor_to_image(style_map[bi].unsqueeze(0).cpu(), vis_dir / f"{prefix}_style_sim_map.png")
                utils.save_tensor_to_image(content_map[bi].unsqueeze(0).cpu(), vis_dir / f"{prefix}_content_sim_map.png")
                saved += 1
    mean_acc_s = total_s / total
    mean_acc_c = total_c / total
    mean_acc = 0.5 * (mean_acc_s + mean_acc_c)
    mean_acc, mean_acc_s, mean_acc_c = mean_acc.item(), mean_acc_s.item(), mean_acc_c.item()

    def _f1(tp, fp, fn, tn):
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return f1, precision, recall, [[tn, fp], [fn, tp]]

    f1_s, prec_s, rec_s, cm_s = _f1(tp_s, fp_s, fn_s, tn_s)
    f1_c, prec_c, rec_c, cm_c = _f1(tp_c, fp_c, fn_c, tn_c)
    macro_f1 = 0.5 * (f1_s + f1_c)

    metrics = {
        "macro_f1": macro_f1,
        "style": {"f1": f1_s, "precision": prec_s, "recall": rec_s, "cm": cm_s},
        "content": {"f1": f1_c, "precision": prec_c, "recall": rec_c, "cm": cm_c},
    }
    return mean_acc, mean_acc_s, mean_acc_c, metrics

def load_gen(cfg, weight_path, device):
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    state = torch.load(weight_path, map_location=device, weights_only=False)
    # 허용: state가 dict로 감싸졌거나 gen만 있을 때 모두 대응
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "gen" in state:
        state = state["gen"]
    gen.load_state_dict(state)
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", required=True, help="path to generator weight")
    parser.add_argument("--csv", help="override cfg.dset.val.data_dir with this CSV path")
    parser.add_argument("--vis_dir", default='./com_map', help="지정 시 유사도 히트맵 PNG 저장 디렉터리")
    parser.add_argument("--vis_n", type=int, default=0, help="저장할 샘플 개수(0이면 저장 안 함)")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    
    if args.csv:
        csv_path = args.csv
    else:
        test_cfg = cfg.dset.test
        csv_path = getattr(test_cfg, "data_dir", test_cfg)
        if not csv_path:
            raise ValueError("--csv를 지정하거나 cfg.dset.test.data_dir에 경로를 설정하세요.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = build_transform(cfg)
    _, val_loader = get_img_loader(
        csv_path, transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
    )

    gen = load_gen(cfg, args.weight, device)
    acc, acc_s, acc_c, metrics = evaluate(
        gen,
        val_loader,
        device,
        cfg.threshold_s,
        cfg.threshold_c,
        vis_dir=Path(args.vis_dir) if args.vis_dir else None,
        vis_n=args.vis_n,
        normalize=bool(cfg.dset_aug.normalize),
    )
    print(f"[test] acc {acc*100:.2f}% | acc_s {acc_s*100:.2f}% | acc_c {acc_c*100:.2f}% "
          f"| macro_f1 {metrics['macro_f1']:.3f} | f1_s {metrics['style']['f1']:.3f} | f1_c {metrics['content']['f1']:.3f}")
    print(f"style cm [[tn, fp], [fn, tp]] = {metrics['style']['cm']} | precision = {metrics['style']['precision']} | recall = {metrics['style']['recall']}")
    print(f"content cm [[tn, fp], [fn, tp]] = {metrics['content']['cm']} | precision = {metrics['content']['precision']} | recall = {metrics['content']['recall']}")

if __name__ == "__main__":
    main()
