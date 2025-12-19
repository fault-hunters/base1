import torch
import torch.nn.functional as F

class PairTrainer:
    def __init__(self, gen, optim, logger, device="cuda", w_style=1.0, w_content=1.0, threshold_s = 0.8, threshold_c = 0.8):
        self.gen = gen.to(device).train()
        self.optim = optim
        self.logger = logger
        self.device = device
        self.w_style = w_style
        self.w_content = w_content
        self.threshold_s = threshold_s
        self.threshold_c = threshold_c

    def forward_pair(self, imgA, imgB):
        sA, cA = self.gen.extract_style_content(imgA)
        sB, cB = self.gen.extract_style_content(imgB)
        sim_s = F.cosine_similarity(sA, sB, dim=1).clamp(-1, 1)
        sim_c = F.cosine_similarity(cA, cB, dim=1).clamp(-1, 1)
        sim_s = (sim_s + 1) / 2  # 0~1
        sim_c = (sim_c + 1) / 2
        return sim_s, sim_c

    def train_one_batch(self, batch):
        imgA, imgB, label_s, label_c = batch
        imgA, imgB = imgA.to(self.device), imgB.to(self.device)
        label_s = label_s.to(self.device).view(-1)
        label_c = label_c.to(self.device).view(-1)

        sim_s, sim_c = self.forward_pair(imgA, imgB)
        loss_s = F.binary_cross_entropy(sim_s, label_s)
        loss_c = F.binary_cross_entropy(sim_c, label_c)
        loss = self.w_style * loss_s + self.w_content * loss_c

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        with torch.no_grad():
            pred_s = (sim_s >= self.threshold_s).float()
            pred_c = (sim_c >= self.threshold_c).float()
            acc_s = (pred_s == label_s).float().mean()
            acc_c = (pred_c == label_c).float().mean()
            acc = 0.5 * (acc_s + acc_c)

        return (
            loss.item(),
            loss_s.item(),
            loss_c.item(),
            acc.item(),
            sim_s.detach(),
            sim_c.detach(),
            acc_s.item(),
            acc_c.item(),
        )

    @torch.no_grad()
    def eval_one_batch(self, batch):
        imgA, imgB, label_s, label_c = batch
        imgA, imgB = imgA.to(self.device), imgB.to(self.device)
        label_s = label_s.to(self.device).view(-1)
        label_c = label_c.to(self.device).view(-1)

        sim_s, sim_c = self.forward_pair(imgA, imgB)
        loss_s = F.binary_cross_entropy(sim_s, label_s)
        loss_c = F.binary_cross_entropy(sim_c, label_c)
        loss = self.w_style * loss_s + self.w_content * loss_c
        pred_s = (sim_s >= self.threshold_s).float()
        pred_c = (sim_c >= self.threshold_c).float()
        acc_s = (pred_s == label_s).float().mean()
        acc_c = (pred_c == label_c).float().mean()
        acc = 0.5 * (acc_s + acc_c)
        bs = label_s.size(0)
        return (
            loss.item(),
            loss_s.item(),
            loss_c.item(),
            acc.item(),
            acc_s.item(),
            acc_c.item(),
            bs,
        )
