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

# setup_args_and_config: 동일 구조, work_dir 준비, n_workers 조정
# setup_transforms: Resize -> ToTensor (+ Normalize)

def train(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    gen.apply(weights_init(cfg.init))
    optim_g = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)

    logger = Logger.get(file_path=cfg.work_dir / "log.log", level="info", colorize=True)
    writer = utils.DiskWriter(cfg.work_dir / "check_img", scale=0.5)
    cfg.tb_freq = -1

    trn_transform = transforms.Compose([
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

    global_step = 0
    for epoch in range(10**9):
        for batch in trn_loader:
            imgA, imgB, label_s, label_c = batch  # 샘플 저장용으로 언팩
            loss, acc, sim_s, sim_c, acc_s, acc_c = trainer.train_one_batch(
                (imgA, imgB, label_s, label_c)
            )

            if global_step % img_freq == 0:
                grid = make_comparable_grid(imgA[:4].cpu(), imgB[:4].cpu(), nrow=4)
                writer.add_image("train_pairs", grid, global_step)
                logger.info(
                    f"step {global_step} | loss {loss:.4f} | acc {acc*100:.2f}% "
                    f"| acc_s {acc_s*100:.2f}% | acc_c {acc_c*100:.2f}% "
                    f"| sim_s {sim_s.mean():.3f} | sim_c {sim_c.mean():.3f}"
                )

            if global_step % cfg.val_freq == 0:
                gen.eval()
                with torch.no_grad():
                    acc_s_list, acc_c_list = [], []
                    for vbatch in val_loader:
                        _, acc_s_v, acc_c_v = trainer.eval_one_batch(vbatch)
                        acc_s_list.append(acc_s_v)
                        acc_c_list.append(acc_c_v)
                    mean_acc_s = sum(acc_s_list)/len(acc_s_list)
                    mean_acc_c = sum(acc_c_list)/len(acc_c_list)
                    mean_acc = 0.5*(mean_acc_s + mean_acc_c)
                    logger.info(
                        f"[val] step {global_step} | acc {mean_acc*100:.2f}% "
                        f"| acc_s {mean_acc_s*100:.2f}% | acc_c {mean_acc_c*100:.2f}%"
                    )
                gen.train()


            if global_step >= cfg.max_iter:
                return
            global_step += 1
