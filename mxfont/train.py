from models import Generator
from models.modules import weights_init
from trainer.pair_trainer import PairTrainer
from datasets_img import get_img_loader
import torch, torch.optim as optim
from torchvision import transforms
import utils
from utils import Logger
import numpy as np
from utils.visualize import make_comparable_grid
from pathlib import Path
import argparse
from sconf import Config

# setup_args_and_config: 동일 구조, work_dir 준비, n_workers 조정
# setup_transforms: Resize -> ToTensor (+ Normalize)

def train(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    optim_g = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device)
        gen.load_state_dict(state.get("gen", state))
        if "optim_g" in state:
            optim_g.load_state_dict(state["optim_g"])
        global_step = state.get("step", 0) + 1  # 위 설명 참고
        if global_step >= cfg.max_iter: return
    else:
        gen.apply(weights_init(cfg.init))
        global_step = 1

    

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get(file_path=cfg.work_dir / "log.log", level="info", colorize=True)
    writer = utils.DiskWriter(cfg.work_dir / "check_img", scale=0.5)
    cfg.tb_freq = -1

    trn_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(cfg.dset_aug.rotation_deg, fill=0)], p=cfg.dset_aug.rotation_p),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) if cfg.dset_aug.normalize else lambda x: x,
    ])
    trn_dset, trn_loader = get_img_loader(
        cfg.dset.train.data_dir, trn_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=True,
    )

    val_dset, val_loader = get_img_loader(
        cfg.dset.val.data_dir, trn_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
    )

    trainer = PairTrainer(
        gen, optim_g, logger, device=device,
        w_style=cfg.get("w_style", 1.0),
        w_content=cfg.get("w_content", 1.0),
        threshold=cfg.threshold,
    )

    img_freq = getattr(cfg, "img_freq", 1000)

    for epoch in range(cfg.epoch):
        for batch in trn_loader:
            imgA, imgB, label_s, label_c = batch  # 샘플 저장용으로 언팩
            loss, loss_s, loss_c, acc, sim_s, sim_c, acc_s, acc_c = trainer.train_one_batch(
                (imgA, imgB, label_s, label_c)
            )

            if global_step % img_freq == 0:
                grid = make_comparable_grid(imgA[:4].cpu(), imgB[:4].cpu(), nrow=4)
                writer.add_image("train_pairs", grid, global_step)
                logger.info(
                    f"[epoch {epoch+1}] step {global_step} | loss {loss:.4f} "
                    f"| loss_s {loss_s:.4f} | loss_c {loss_c:.4f} | acc {acc*100:.2f}% "
                    f"| acc_s {acc_s*100:.2f}% | acc_c {acc_c*100:.2f}% "
                    f"| sim_s {sim_s.mean().item():.3f} | sim_c {sim_c.mean().item():.3f}"
                )

            if (global_step % cfg.val_freq == 0) and (global_step > 0):
                gen.eval()
                total_loss = total_loss_s = total_loss_c = total_s = total_c = total = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        loss_v, loss_s_v, loss_c_v, acc_v, acc_s_v, acc_c_v, bs = trainer.eval_one_batch(vbatch)
                        total_loss += loss_v * bs
                        total_loss_s += loss_s_v * bs
                        total_loss_c += loss_c_v * bs
                        total_s += acc_s_v * bs
                        total_c += acc_c_v * bs
                        total += bs
                    mean_loss = total_loss / total
                    mean_loss_s = total_loss_s / total
                    mean_loss_c = total_loss_c / total
                    mean_acc_s = total_s / total
                    mean_acc_c = total_c / total
                    mean_acc = 0.5 * (mean_acc_s + mean_acc_c)
                    logger.info(
                        f"[val] step {global_step} | loss {mean_loss:.4f} "
                        f"| loss_s {mean_loss_s:.4f} | loss_c {mean_loss_c:.4f} "
                        f"| acc {mean_acc*100:.2f}% "
                        f"| acc_s {mean_acc_s*100:.2f}% | acc_c {mean_acc_c*100:.2f}%"
                    )
                gen.train()

            if (global_step % cfg.save_freq == 0) or (global_step >= cfg.max_iter):
                torch.save(
                    {"gen": gen.state_dict(), "optim_g": optim_g.state_dict(), "step": global_step, "cfg": cfg},
                    cfg.work_dir / f"gen_{global_step}.pth"
                )

            
            if global_step >= cfg.max_iter:
                return
            global_step += 1

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    args, left = parser.parse_known_args()
    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left)
    return args, cfg

if __name__ == "__main__":
    args, cfg = parse_cfg()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train(args, cfg)
