# train.py
import os, re, argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import Counter, defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from modules import build_model
from dataset import get_loaders, set_seed

# ----------------- Global controls -----------------
metric_for_search = "bal_acc"   # 可选: "acc" | "bal_acc" | "youden"
target_tpr = 0.7                # 例如 0.75；不设则为 None

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

def _pick_threshold_with_target_tpr(sid_true, sid_probs, target_tpr=None, metric="bal_acc"):
    """
    在 subject-level 上选阈值：
      - 若 target_tpr 非空：只在 TPR >= target_tpr 的阈值候选中最大化 metric
      - 若没有阈值能满足 target_tpr，则退化为在所有阈值里最大化 metric
    返回 best_thr
    """
    thrs = np.linspace(0.0, 1.0, 2001)

    def _tpr_tnr(pred, true):
        tp = ((pred == 1) & (true == 1)).sum()
        fn = ((pred == 0) & (true == 1)).sum()
        tn = ((pred == 0) & (true == 0)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        return float(tpr), float(tnr)

    best_thr, best_metric = 0.5, -1.0
    cand = []
    for t in thrs:
        pred = (sid_probs > t).astype(int)
        tpr, tnr = _tpr_tnr(pred, sid_true)
        if metric == "bal_acc":
            m = 0.5 * (tpr + tnr)
        elif metric == "youden":
            m = tpr + tnr - 1.0
        else:  # "acc"
            m = accuracy_score(sid_true, pred)
        cand.append((t, m, tpr))

    if target_tpr is not None:
        hit = [x for x in cand if x[2] + 1e-12 >= target_tpr]
        if hit:
            for t, m, _ in hit:
                if m > best_metric:
                    best_metric, best_thr = m, float(t)
        else:
            for t, m, _ in cand:
                if m > best_metric:
                    best_metric, best_thr = m, float(t)
    else:
        for t, m, _ in cand:
            if m > best_metric:
                best_metric, best_thr = m, float(t)

    return best_thr

@torch.no_grad()
def evaluate_subject_level_split(
    model,
    loader,
    criterion,
    device,
    threshold=None,
    search_thresh=False,
    metric_for_search="acc"  # 可传入覆盖全局设定；不传就用这里的默认或全局的
):
    """
    返回: epoch_loss, acc_overall, auc_overall, best_thr, per_class
    per_class = {
        "AD_n": int, "NC_n": int,
        "AD_acc": float, "NC_acc": float,
        "TP": int, "TN": int, "FP": int, "FN": int,
        "recall_AD": float,     # AD召回(敏感度)
        "specificity_NC": float # NC特异度
    }
    """
    model.eval()
    running_loss, y_true_img, y_prob_img = 0.0, [], []

    paths_all = _ordered_paths_from_loader(loader)
    ptr = 0
    paths_this_epoch = []

    for imgs, labels in loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.detach().item() * bs

        # 概率定义：这里取“类1(NC)”的概率；因此 (prob > thr)->预测NC
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs])
        ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

    # ---- 聚合到 subject 级（均值；也可以换成 median）----
    prob_by_sid = defaultdict(list)
    label_by_sid = {}
    for prob, y, p in zip(y_prob_img, y_true_img, paths_this_epoch):
        sid = _extract_subject_id(p)
        prob_by_sid[sid].append(prob)
        label_by_sid.setdefault(sid, []).append(int(y))

    sid_list = sorted(prob_by_sid.keys())
    sid_probs = np.array([np.mean(prob_by_sid[s]) for s in sid_list], dtype=np.float32)
    sid_true  = np.array([int(round(np.mean(label_by_sid[s]))) for s in sid_list], dtype=np.int64)

    # ---- 选阈值（按 subject 级）----
    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        _metric = metric_for_search if metric_for_search is not None else globals().get("metric_for_search", "acc")
        _target_tpr = globals().get("target_tpr", None)
        best_thr = _pick_threshold_with_target_tpr(
            sid_true, sid_probs,
            target_tpr=_target_tpr,
            metric=_metric
        )

    # ---- 应用阈值并计算指标 ----
    sid_pred = (sid_probs > best_thr).astype(int)

    acc_overall = accuracy_score(sid_true, sid_pred)
    try:
        auc_overall = roc_auc_score(sid_true, sid_probs)
    except Exception:
        auc_overall = 0.0

    # 分类别统计（把 AD=0 当“正类”来汇报更方便看召回）
    mask_AD = (sid_true == 0)
    mask_NC = (sid_true == 1)
    AD_n = int(mask_AD.sum())
    NC_n = int(mask_NC.sum())

    AD_acc = float(((sid_pred == 0) & mask_AD).sum() / AD_n) if AD_n > 0 else float("nan")
    NC_acc = float(((sid_pred == 1) & mask_NC).sum() / NC_n) if NC_n > 0 else float("nan")

    TP = int(((sid_pred == 0) & (sid_true == 0)).sum())  # 真实AD且预测AD
    TN = int(((sid_pred == 1) & (sid_true == 1)).sum())  # 真实NC且预测NC
    FP = int(((sid_pred == 0) & (sid_true == 1)).sum())  # 真实NC但预测AD
    FN = int(((sid_pred == 1) & (sid_true == 0)).sum())  # 真实AD但预测NC

    per_class = {
        "AD_n": AD_n, "NC_n": NC_n,
        "AD_acc": AD_acc, "NC_acc": NC_acc,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "recall_AD": float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan"),
        "specificity_NC": float(TN / (TN + FP)) if (TN + FP) > 0 else float("nan"),
    }

    return float(epoch_loss), float(acc_overall), float(auc_overall), float(best_thr), per_class

@torch.no_grad()
def evaluate_subject_level(model, loader, criterion, device,
                           threshold=None, search_thresh=False):
    model.eval()
    running_loss, y_true_img, y_prob_img = 0.0, [], []

    paths_all = _ordered_paths_from_loader(loader)
    ptr = 0
    paths_this_epoch = []

    for imgs, labels in loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.detach().item() * bs

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs])
        ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

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

    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        best_thr = _pick_threshold_with_target_tpr(
            sid_true, sid_probs,
            target_tpr=target_tpr,
            metric=metric_for_search
        )

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

def _collect_labels(ds):
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
        p.requires_grad_((False))

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

    # Warmup + Cosine
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

    scaler = GradScaler('cuda')
    ema_decay = 0.999  # 固定 EMA 衰减

    # ------- Early Stopping setup -------
    patience = args.early_stop_patience
    min_delta = args.early_stop_delta
    epochs_no_improve = 0

    # Train loop
    best_val_auc = -1.0
    best_thr = 0.5

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "val_auc": [],
        "test_acc": [], "test_auc": [],
        "lr": [], "best_thr": []
    }

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # step LR
        cur_lr = compute_lr(epoch)
        set_lr(optimizer, cur_lr)

        # ---- Train ----
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            ema_model=ema_model, ema_decay=ema_decay
        )

        # ---- Val（患者级，含阈值搜索）----
        val_loss, val_acc, val_auc, best_thr_epoch = evaluate_subject_level(
            ema_model, val_loader, criterion, device, search_thresh=True
        )

        # ---- Test（患者级，用 Val 的最佳阈值）----
        test_loss, test_acc, test_auc, _, test_split = evaluate_subject_level_split(
            ema_model, test_loader, criterion, device,
            threshold=best_thr_epoch, search_thresh=False
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
        print(f"Test  Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"Per-class Test Acc -> AD: {test_split['AD_acc']:.4f} (n={test_split['AD_n']}), "
              f"NC: {test_split['NC_acc']:.4f} (n={test_split['NC_n']})")
        print(f"Confusion (AD positive): TP={test_split['TP']}  FP={test_split['FP']}  "
              f"TN={test_split['TN']}  FN={test_split['FN']}")
        print(f"Recall_AD={test_split['recall_AD']:.4f}, Specificity_NC={test_split['specificity_NC']:.4f}")
        print(f"LR after epoch {epoch+1}: {cur_lr:.6f}")

        # ------- Save best & Early stopping update -------
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_thr = best_thr_epoch
            torch.save(
                {"model": ema_model.state_dict(),
                 "args": vars(args),
                 "class_names": class_names,
                 "best_thr": best_thr},
                os.path.join(args.outdir, "best_model.pt")
            )
            print("✅ Saved best EMA model (by subject-level AUC)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"EarlyStop watch: no improve for {epochs_no_improve}/{patience} epochs "
                  f"(need > {min_delta} gain)")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without sufficient improvement.")
            break

    # -------- Final test with the best checkpoint --------
    ck = torch.load(os.path.join(args.outdir, "best_model.pt"), map_location=device)
    ema_model.load_state_dict(ck["model"])
    best_thr_final = float(ck.get("best_thr", best_thr))

    test_loss, test_acc, test_auc, _ = evaluate_subject_level(
        ema_model, test_loader, criterion, device,
        threshold=best_thr_final, search_thresh=False
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
