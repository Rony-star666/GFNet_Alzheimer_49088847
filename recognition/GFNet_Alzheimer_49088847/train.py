# train.py
import os, re, argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from modules import build_model
from dataset import get_loaders, set_seed


# ----------------- Temperature Scaling -----------------
class _TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))  # T=1 起步
    def forward(self, logits):
        T = torch.exp(self.logT) + 1e-6
        return logits / T

@torch.no_grad()
def _collect_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.clone())
    return torch.cat(all_logits), torch.cat(all_labels)

def fit_temperature(model, val_loader, device):
    logits, labels = _collect_logits_labels(model, val_loader, device)
    temp = _TempScale().to(device)
    logits = logits.to(device)
    labels = labels.to(device)
    opt = torch.optim.LBFGS(temp.parameters(), lr=0.1, max_iter=50)

    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = ce(temp(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return temp

def _with_temp_forward(model, temp_layer, imgs):
    logits = model(imgs)
    return temp_layer(logits) if temp_layer is not None else logits


# ----------------- AutoTuner -----------------
class AutoTuner:
    """
    每个 epoch 后，根据 train/val 的差异自适应地做小幅正则化调整。
    """
    def __init__(self, model, optimizer, criterion, ema=None,
                 gap_hi=0.08, gap_lo=0.02, patience=3,
                 wd_bounds=(1e-4, 5e-2), ls_bounds=(0.0, 0.10),
                 dp_bounds=(0.0, 0.4)):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ema = ema
        self.gap_hi = gap_hi
        self.gap_lo = gap_lo
        self.patience = patience
        self.wd_bounds = wd_bounds
        self.ls_bounds = ls_bounds
        self.dp_bounds = dp_bounds

        self.best_val = -1.0       # ✅ 用成员变量
        self.bad_epochs = 0

        self.bad_epochs = 0

    def _set_weight_decay(self, new_wd):
        new_wd = float(min(max(new_wd, self.wd_bounds[0]), self.wd_bounds[1]))
        for g in self.optimizer.param_groups:
            if g.get("weight_decay", 0.0) > 0:
                g["weight_decay"] = new_wd

    def _get_weight_decay(self):
        for g in self.optimizer.param_groups:
            if g.get("weight_decay", 0.0) > 0:
                return g["weight_decay"]
        return 0.0

    def _set_label_smoothing(self, new_ls):
        new_ls = float(min(max(new_ls, self.ls_bounds[0]), self.ls_bounds[1]))
        if hasattr(self.criterion, "label_smoothing"):
            self.criterion.label_smoothing = new_ls

    def _get_label_smoothing(self):
        return getattr(self.criterion, "label_smoothing", 0.0)

    def _set_dropout(self, new_p):
        new_p = float(min(max(new_p, self.dp_bounds[0]), self.dp_bounds[1]))
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.p = new_p

    def _get_dropout(self):
        ps = []
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                ps.append(m.p)
        return sum(ps)/len(ps) if ps else 0.0

    def step(self, tr_acc, val_acc, val_auc, improve, tf_train=None):
        # 这里的 improve 由调用方决定（本脚本用 val_acc 判断）
        if improve:
            self.best_val = max(self.best_val, val_auc)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        gap = float(tr_acc - val_acc)
        wd = self._get_weight_decay()
        ls = self._get_label_smoothing()
        dp = self._get_dropout()

        # 过拟合：gap 大且无提升 -> 增强正则
        if gap > self.gap_hi and self.bad_epochs >= 1:
            target = wd * 1.5 if wd > 0 else self.wd_bounds[0]
            self._set_weight_decay(target)
            self._set_label_smoothing(ls + 0.01)
            self._set_dropout(dp + 0.02)
            if self.ema is not None and isinstance(self.ema, dict):
                self.ema["decay"] = min(0.9999, self.ema["decay"] + 0.0005)

        # 欠拟合：gap 小且 val 低 -> 降正则
        elif gap < self.gap_lo and val_acc < 0.65 and self.bad_epochs >= 2:
            self._set_weight_decay(wd * 0.7)
            self._set_label_smoothing(max(self.ls_bounds[0], ls - 0.01))
            self._set_dropout(max(self.dp_bounds[0], dp - 0.02))

        return {
            "wd": self._get_weight_decay(),
            "ls": self._get_label_smoothing(),
            "dp": self._get_dropout(),
            "bad_epochs": self.bad_epochs,
            "gap": gap
        }


# ----------------- EMA -----------------
def ema_update(ema_model, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in msd.keys():
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)


# ----------------- Train one epoch -----------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                    ema_model=None, ema_decay=0.999):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_update(ema_model, model, decay=ema_decay)

        total_loss += loss.detach().item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ----------------- Subject-level evaluation -----------------
def _ordered_paths_from_loader(loader):
    ds = loader.dataset
    paths = []
    if hasattr(ds, "indices") and hasattr(ds, "dataset") and hasattr(ds.dataset, "samples"):
        for idx in ds.indices:
            paths.append(ds.dataset.samples[idx][0])
    elif hasattr(ds, "samples"):
        for p, _ in ds.samples:
            paths.append(p)
    else:
        raise RuntimeError("无法从当前 dataset 恢复路径；请确保 val/test 的 DataLoader shuffle=False。")
    return paths

def _extract_subject_id(p):
    m = re.findall(r"\d{5,}", p.replace("\\", "/"))
    if m:
        return max(m, key=len)  # 取最长数字串
    stem = os.path.splitext(os.path.basename(p))[0]
    return stem.split("_")[0]

@torch.no_grad()
def evaluate_subject_level(model, loader, criterion, device,
                           threshold=None, search_thresh=False, temp_layer=None):
    model.eval()
    running_loss, y_true_img, y_prob_img = 0.0, [], []
    metric = "bal_acc"  # 用 balanced accuracy 搜阈值

    # 注意：val/test loader 要 shuffle=False 才能用这个函数
    paths_all = _ordered_paths_from_loader(loader)
    ptr = 0
    paths_this_epoch = []

    for imgs, labels in loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        logits = _with_temp_forward(model, temp_layer, imgs)
        loss = criterion(logits, labels)
        running_loss += loss.detach().item() * bs

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs])
        ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

    # ---- 患者级聚合：用 median 抗噪（也可以换成截尾平均）----
    from statistics import median
    prob_by_sid = defaultdict(list)
    label_by_sid = {}
    for prob, y, p in zip(y_prob_img, y_true_img, paths_this_epoch):
        sid = _extract_subject_id(p)
        prob_by_sid[sid].append(prob)
        label_by_sid.setdefault(sid, [])
        label_by_sid[sid].append(int(y))

    sid_list = sorted(prob_by_sid.keys())
    sid_probs = np.array([median(prob_by_sid[s]) for s in sid_list], dtype=np.float32)
    sid_true  = np.array([int(round(np.mean(label_by_sid[s]))) for s in sid_list], dtype=np.int64)

    # ---- 阈值 ----
    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        thrs = np.linspace(0.0, 1.0, 201)
        best_metric = -1.0
        for t in thrs:
            pred = (sid_probs > t).astype(int)
            m = (balanced_accuracy_score(sid_true, pred)
                 if metric == "bal_acc" else
                 accuracy_score(sid_true, pred))
            if m > best_metric:
                best_metric = m
                best_thr = float(t)

    # 报告的 acc 仍用普通 accuracy，便于与历史曲线对齐
    acc = accuracy_score(sid_true, (sid_probs > best_thr).astype(int))

    try:
        auc = roc_auc_score(sid_true, sid_probs)
    except Exception:
        auc = 0.0

    return float(epoch_loss), float(acc), float(auc), float(best_thr)



# ----------------- Plot utils -----------------
def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 1) Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()

    # 2) Accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"],   label="val_acc")
    if "test_acc" in history and len(history["test_acc"]) == len(history["val_acc"]):
        plt.plot(history["test_acc"],  label="test_acc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png")); plt.close()

    # 3) AUC
    plt.figure()
    plt.plot(history["val_auc"],  label="val_auc")
    if "test_auc" in history and len(history["test_auc"]) == len(history["val_auc"]):
        plt.plot(history["test_auc"], label="test_auc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("AUC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png")); plt.close()

    # 4) Learning rate
    if "lr" in history and len(history["lr"]) > 0:
        plt.figure()
        plt.plot(history["lr"], label="lr")
        plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lr_curve.png")); plt.close()

    # 5) Best threshold (subject-level)
    if "best_thr" in history and len(history["best_thr"]) > 0:
        plt.figure()
        plt.plot(history["best_thr"], label="best_thr (val subject-level)")
        plt.xlabel("epoch"); plt.ylabel("threshold"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_curve.png")); plt.close()

    # 6) AutoTuner 正则轨迹
    if "wd" in history and len(history["wd"]) > 0:
        plt.figure()
        plt.plot(history["wd"], label="weight_decay")
        plt.xlabel("epoch"); plt.ylabel("weight_decay"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "wd_curve.png")); plt.close()

    if "ls" in history and len(history["ls"]) > 0:
        plt.figure()
        plt.plot(history["ls"], label="label_smoothing")
        plt.xlabel("epoch"); plt.ylabel("label_smoothing"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "label_smoothing_curve.png")); plt.close()

    if "dp" in history and len(history["dp"]) > 0:
        plt.figure()
        plt.plot(history["dp"], label="dropout_p")
        plt.xlabel("epoch"); plt.ylabel("dropout"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "dropout_curve.png")); plt.close()


def _collect_labels(ds):
    # 兼容 ImageFolder / 自定义
    if hasattr(ds, 'targets'):
        return list(map(int, ds.targets))
    if hasattr(ds, 'samples'):
        labs = []
        for s in ds.samples:
            _, cls = s
            labs.append(int(cls))
        return labs
    return None

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


# ----------------- Main -----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Loaders
    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        gray=(args.in_channels == 1)
    )

    # Class weights
    labs = _collect_labels(train_loader.dataset)
    class_weights = None
    if labs:
        cnt = Counter(labs)
        total = sum(cnt.values())
        w0 = total / (2.0 * max(cnt.get(0, 1), 1))
        w1 = total / (2.0 * max(cnt.get(1, 1), 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Class weights -> class0: {w0:.4f}, class1: {w1:.4f}")

    # Model & EMA
    model = build_model(in_channels=args.in_channels, height=args.img_size, width=args.img_size).to(device)
    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    # Optimizer (decoupled decay/no_decay)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr
    )

    # Warmup + Cosine（手工调 lr，避免与 EMA 打架）
    base_lr = args.lr
    min_lr = 1e-6
    warmup_epochs = 3

    def compute_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        # cosine from base_lr -> min_lr
        t = epoch - warmup_epochs
        T = max(1, args.epochs - warmup_epochs)
        import math
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))

    # AutoTuner
    ema_decay_ref = {"decay": 0.999}
    tuner = AutoTuner(
        model=model, optimizer=optimizer, criterion=criterion, ema=ema_decay_ref,
        gap_hi=0.08, gap_lo=0.02, patience=3,
        wd_bounds=(1e-4, 5e-2), ls_bounds=(0.0, 0.10), dp_bounds=(0.0, 0.4)
    )
    scaler = GradScaler('cuda')

    # Train loop
    best_val_auc = -1.0    # ✅ 初始化 AUC 基线
    best_thr = 0.5

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "val_auc": [],
        "test_acc": [], "test_auc": [],
        "lr": [], "best_thr": [],
        "wd": [], "ls": [], "dp": []
    }

    os.makedirs(args.outdir, exist_ok=True)

    temp_layer = None
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # step LR
        cur_lr = compute_lr(epoch)
        set_lr(optimizer, cur_lr)

        # ---- Train ----
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            ema_model=ema_model, ema_decay=ema_decay_ref["decay"]
        )

        # ---- 温度标定（用验证集；不要用 test）----
        if epoch == 0 or (epoch % 5 == 0):
            temp_layer = fit_temperature(ema_model, val_loader, device)
            for p in temp_layer.parameters():
                p.requires_grad_(False)

        # ---- Val（患者级+温度）----
        val_loss, val_acc, val_auc, best_thr_epoch = evaluate_subject_level(
            ema_model, val_loader, criterion, device, search_thresh=True, temp_layer=temp_layer
        )

        # ---- Test（患者级+温度，用 Val 的最佳阈值）----
        test_loss, test_acc, test_auc, _ = evaluate_subject_level(
            ema_model, test_loader, criterion, device,
            threshold=best_thr_epoch, search_thresh=False, temp_layer=temp_layer
        )

        # 记录
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["test_acc"].append(test_acc)
        history["test_auc"].append(test_auc)
        history["lr"].append(cur_lr)
        history["best_thr"].append(best_thr_epoch)

        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, BestThr: {best_thr_epoch:.3f}")
        print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"LR after epoch {epoch+1}: {cur_lr:.6f}")

        # AutoTuner（用 AUC 作为提升判断）
        eps = 1e-6
        improve = (val_auc > best_val_auc + eps)   # ✅ 用 AUC 判是否提升
        info = tuner.step(tr_acc, val_acc, val_auc, improve=improve)

        history["wd"].append(info["wd"])
        history["ls"].append(info["ls"])
        history["dp"].append(info["dp"])
        print(f"[AutoTuner] wd={info['wd']:.2e} ls={info['ls']:.3f} dp={info['dp']:.2f} "
              f"gap={info['gap']:.3f} bad_epochs={info['bad_epochs']} (ema_decay={ema_decay_ref['decay']:.5f})")

        # 保存：按 val_auc 最优
        if val_auc > best_val_auc + eps:
            best_val_auc = val_auc              # ✅ 更新“最佳 AUC”
            best_thr = best_thr_epoch
            torch.save(
                {   "model": ema_model.state_dict(),
                    "args": vars(args),
                    "class_names": class_names,
                    "best_thr": best_thr
                },
                os.path.join(args.outdir, "best_model.pt")
            )
            print("✅ Saved best EMA model (by subject-level AUC)")



    # -------- Final test with the best checkpoint --------
    ck = torch.load(os.path.join(args.outdir, "best_model.pt"), map_location=device)
    ema_model.load_state_dict(ck["model"])
    best_thr_final = float(ck.get("best_thr", best_thr))

    # 再用验证集拟合一次温度，保持一致性
    temp_layer = fit_temperature(ema_model, val_loader, device)
    for p in temp_layer.parameters():
        p.requires_grad_(False)

    test_loss, test_acc, test_auc, _ = evaluate_subject_level(
        ema_model, test_loader, criterion, device,
        threshold=best_thr_final, search_thresh=False, temp_layer=temp_layer
    )
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f} (thr={best_thr_final:.3f})")

    plot_curves(history, args.outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ADNI/AD_NC")
    ap.add_argument("--outdir", type=str, default="runs/adni_gfnet")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
