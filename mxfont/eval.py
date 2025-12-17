import argparse
from pathlib import Path
import torch
from torchvision import transforms
from sconf import Config
from models import Generator
from datasets_img import get_img_loader

def build_transform(cfg):
    return transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(cfg.dset_aug.rotation_deg, fill=0)], p=cfg.dset_aug.rotation_p),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) if cfg.dset_aug.normalize else lambda x: x,
    ])

@torch.no_grad()
def evaluate(gen, loader, device, threshold):
    gen.eval()
    total_s = total_c = total = 0
    tp_s = fp_s = fn_s = tn_s = 0
    tp_c = fp_c = fn_c = tn_c = 0
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

        pred_s = (sim_s >= threshold).float()
        pred_c = (sim_c >= threshold).float()
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
    acc, acc_s, acc_c, metrics = evaluate(gen, val_loader, device, cfg.threshold)
    print(f"[test] acc {acc*100:.2f}% | acc_s {acc_s*100:.2f}% | acc_c {acc_c*100:.2f}% "
          f"| macro_f1 {metrics['macro_f1']:.3f} | f1_s {metrics['style']['f1']:.3f} | f1_c {metrics['content']['f1']:.3f}")
    print(f"style cm [[tn, fp], [fn, tp]] = {metrics['style']['cm']}")
    print(f"content cm [[tn, fp], [fn, tp]] = {metrics['content']['cm']}")

if __name__ == "__main__":
    main()
