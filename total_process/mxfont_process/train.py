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
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sconf import Config
import torch.multiprocessing as mp
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

# setup_args_and_config: 동일 구조, work_dir 준비, n_workers 조정
# setup_transforms: Resize -> ToTensor (+ Normalize)

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


def cleanup():
    dist.destroy_process_group()


def is_main_worker(gpu):
    return (gpu <= 0)


def train_ddp(gpu, args, cfg, world_size):
    print(f"[rank{gpu}] enter train_ddp (pid={os.getpid()})", flush=True)
    print(f"[rank{gpu}] before init_process_group", flush=True)
    print(f"[rank{gpu}] after init_process_group", flush=True)
    dist.init_process_group(backend="nccl",
                            init_method="tcp://127.0.0.1:" + str(cfg.port),
                            world_size=world_size,
                            rank=gpu,)
    print(f"[rank{gpu}] after init_process_group", flush=True)
    try:
        train(args, cfg, ddp_gpu=gpu)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        print(f"[rank{gpu}] destroyed process group", flush=True)
        #cleanup()

def train(args, cfg, ddp_gpu):
    print(f"[rank{ddp_gpu}] enter train", flush=True)
    if cfg.use_ddp:
        torch.cuda.set_device(ddp_gpu)
        device = torch.device(f"cuda:{ddp_gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[rank{ddp_gpu}] device={device}", flush=True)
    gen = Generator(3, cfg.C, 1, **cfg.get("g_args", {})).to(device)
    print(f"[rank{ddp_gpu}] generator created", flush=True)

    if cfg.use_ddp:
        gen = DDP(gen, device_ids=[ddp_gpu], output_device=ddp_gpu)
        print(f"[rank{ddp_gpu}] wrapped with DDP", flush=True)
    
    optim_g = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)

    ################## gen state train ##################
    if cfg.split:
        state_c = torch.load(cfg.resume_c, map_location=device, weights_only=False)
        state_s = torch.load(cfg.resume_s, map_location=device, weights_only=False)
        # g.style_enc, g.experts_s, g.fuser_style, g.fact_blocks_s
        new_state = {}
        for k in state_s.keys():
            if k.startswith("style_enc") or k.startswith("experts_s") or k.startswith("fuser_style") or k.startswith("fact_blocks_s"):
                new_state[k] = state_s[k]
            else:
                new_state[k] = state_c[k]
        global_step = 1
        gen.load_state_dict(new_state, strict=True)

    elif cfg.resume:
        print(cfg.resume)
        state = torch.load(cfg.resume, map_location=device, weights_only=False)
        gen.load_state_dict(state.get("gen", state))
        if "optim_g" in state:
            optim_g.load_state_dict(state["optim_g"])
        
            for param_group in optim_g.param_groups:
                print("learning rate : ", param_group["lr"])
                #param_group["lr"] = cfg.g_lr # 학습률 바꿀 때
        global_step = state.get("step", 0) + 1
        epoch = state.get("epoch", 0) + 1
        if epoch >= cfg.epoch: return
    else:
        print("No Weight")
        gen.apply(weights_init(cfg.init))
        global_step = 1
    #######################################################   

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger.get(file_path=cfg.work_dir / "log.log", level="info", colorize=True)
    writer = utils.DiskWriter(cfg.work_dir / "check_img", scale=0.5)
    cfg.tb_freq = -1
    print(f"[rank{ddp_gpu}] build transforms", flush=True)
    trn_transform = transforms.Compose([
        SquarePad(fill=(255, 255, 255)),
        transforms.Resize((1024, 1024)), # input img resizing 512X512
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) if cfg.dset_aug.normalize else lambda x: x,
    ])
    print(f"[rank{ddp_gpu}] before get_img_loader(train)", flush=True)
    trn_dset, trn_loader = get_img_loader(
        cfg.dset.train.data_dir, cfg.use_ddp, trn_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=True,
    )
    print(f"[rank{ddp_gpu}] after get_img_loader(train) len={len(trn_dset)}", flush=True)

    print(f"[rank{ddp_gpu}] before get_img_loader(val)", flush=True)
    val_dset, val_loader = get_img_loader(
        cfg.dset.val.data_dir, cfg.use_ddp, trn_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
    )
    print(f"[rank{ddp_gpu}] after get_img_loader(val) len={len(val_dset)}", flush=True)
    trainer = PairTrainer(
        gen, optim_g, cfg, logger, device=device,
        w_style=cfg.get("w_style", 0.5),
        w_content=cfg.get("w_content", 0.5),
        threshold_s=cfg.threshold_s,
        threshold_c=cfg.threshold_c,
        
    )
    print(f"[rank{ddp_gpu}] before first batch", flush=True)
    it = iter(trn_loader)
    batch = next(it)
    print(f"[rank{ddp_gpu}] got first batch", flush=True)

    img_freq = getattr(cfg, "img_freq", 1000)
    print(f"[rank{ddp_gpu}] start train", flush=True)

    for epoch in range(cfg.epoch):
        if cfg.use_ddp and hasattr(trn_loader, "sampler"):
            trn_loader.sampler.set_epoch(epoch)
        if cfg.use_ddp and hasattr(val_loader, "sampler"):
            val_loader.sampler.set_epoch(epoch)
        
        is_main = is_main_worker(ddp_gpu)
        pbar = tqdm(
            trn_loader,
            total=len(trn_loader),
            desc=f"epoch {epoch}",
            disable=not is_main,
            dynamic_ncols=True,
        )

        train_acc_s = train_acc_c = train_loss = train_loss_s = train_loss_c = train_total = 0
        for batch in pbar:
            imgA, imgB, label_s, label_c = batch  # 샘플 저장용으로 언팩
            loss, loss_s, loss_c, acc, sim_s, sim_c, acc_s, acc_c, bs = trainer.train_one_batch(
                (imgA, imgB, label_s, label_c)
            )
            train_acc_s += acc_s * bs
            train_acc_c += acc_c * bs
            train_loss += loss * bs
            train_loss_s += loss_s * bs
            train_loss_c += loss_c * bs
            train_total += bs
            mean_loss = train_loss / train_total
            mean_loss_s = train_loss_s / train_total
            mean_loss_c = train_loss_c / train_total
            mean_acc_s = train_acc_s / train_total
            mean_acc_c = train_acc_c / train_total
            mean_acc = 0.5 * (mean_acc_s + mean_acc_c)
            pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                acc=f"{mean_acc*100:.2f}%",
            )

            global_step += 1

        if cfg.use_ddp:
            totals = torch.tensor(
                [train_acc_s, train_acc_c, train_loss, train_loss_s, train_loss_c, train_total],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            train_acc_s, train_acc_c, train_loss, train_loss_s, train_loss_c, train_total = totals.tolist()
        mean_loss = train_loss / train_total
        mean_loss_s = train_loss_s / train_total
        mean_loss_c = train_loss_c / train_total
        mean_acc_s = train_acc_s / train_total
        mean_acc_c = train_acc_c / train_total
        mean_acc = 0.5 * (mean_acc_s + mean_acc_c)

        if epoch % img_freq == 0 and is_main_worker(ddp_gpu):
            grid = make_comparable_grid(imgA[:4].cpu(), imgB[:4].cpu(), nrow=4)
            writer.add_image("train_pairs", grid, global_step)
            logger.info(
                f"[epoch {epoch}] loss {mean_loss:.4f} "
                f"| loss_s {mean_loss_s:.4f} | loss_c {mean_loss_c:.4f} | acc {mean_acc*100:.2f}% "
                f"| acc_s {mean_acc_s*100:.2f}% | acc_c {mean_acc_c*100:.2f}% "
            )

        if (epoch % cfg.val_freq == 0):
            print(f"[rank{ddp_gpu}] epoch{epoch} validation start", flush=True)
            sim_info = []
            gen.eval()
            total_loss = total_loss_s = total_loss_c = total_s = total_c = total = 0
            with torch.no_grad():
                
                vbar = tqdm(
                    val_loader,
                    total=len(val_loader),
                    desc=f"val {epoch}",
                    disable=not is_main_worker(ddp_gpu),
                    dynamic_ncols=True,
                )
                for vbatch in vbar:
                    sim_s, sim_c, label_s, label_c, loss_v, loss_s_v, loss_c_v, acc_v, acc_s_v, acc_c_v, bs = trainer.eval_one_batch(vbatch)
                    sim_info.append({'sim_s': sim_s, 'sim_c': sim_c, 'label_s': label_s, 'label_c': label_c})
                    total_loss += loss_v * bs
                    total_loss_s += loss_s_v * bs
                    total_loss_c += loss_c_v * bs
                    total_s += acc_s_v * bs
                    total_c += acc_c_v * bs
                    total += bs
                    mean_loss = total_loss / total
                    mean_acc_s = total_s / total
                    mean_acc_c = total_c / total
                    mean_acc = 0.5 * (mean_acc_s + mean_acc_c)
                    vbar.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        acc=f"{mean_acc*100:.2f}%",
                    )
                
                if cfg.use_ddp:
                    val_totals = torch.tensor(
                        [total_s, total_c, total_loss, total_loss_s, total_loss_c, total],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(val_totals, op=dist.ReduceOp.SUM)
                    total_s, total_c, total_loss, total_loss_s, total_loss_c, total = val_totals.tolist()
                
                mean_loss = total_loss / total
                mean_loss_s = total_loss_s / total
                mean_loss_c = total_loss_c / total
                mean_acc_s = total_s / total
                mean_acc_c = total_c / total
                mean_acc = 0.5 * (mean_acc_s + mean_acc_c)
                if is_main_worker(ddp_gpu):
                    logger.info(
                        f"[val] epoch {epoch} | loss {mean_loss:.4f} "
                        f"| loss_s {mean_loss_s:.4f} | loss_c {mean_loss_c:.4f} "
                        f"| acc {mean_acc*100:.2f}% "
                        f"| acc_s {mean_acc_s*100:.2f}% | acc_c {mean_acc_c*100:.2f}%"
                    )
                sim_df = pd.DataFrame(sim_info)
                work_dir = cfg.work_dir
                if not work_dir:
                    work_dir = cfg.work_dir
                csv_path = Path(work_dir) / f"similarity_eval_{epoch}_rank{ddp_gpu}.csv"
                sim_df.to_csv(csv_path, index=False, sep=',', encoding='utf-8')
            gen.train()

        if ((epoch % cfg.save_freq == 0) or (epoch >= cfg.max_iter)) and is_main_worker(ddp_gpu):
            torch.save(
                {"gen": gen.state_dict(), "optim_g": optim_g.state_dict(), "epoch": epoch, "step": global_step, "cfg": cfg},
                cfg.work_dir / f"gen_{epoch}.pth"
            )

        
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    args, left = parser.parse_known_args()
    cfg = Config(*args.config_paths, default="base1/mxfont/cfgs/defaults.yaml")
    cfg.argv_update(left)

    if cfg.use_ddp:
        cfg.n_workers = 0
    cfg.work_dir = Path(cfg.work_dir)
    (cfg.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return args, cfg

if __name__ == "__main__":
    args, cfg = parse_cfg()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.use_ddp:
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)
