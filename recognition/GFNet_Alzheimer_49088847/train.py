# train.py —— no-EMA / no-Temp / no-AutoTuner
import os, re, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from modules import build_model
from dataset import get_loaders, set_seed

# 阈值搜索的全局设定
metric_for_search = "bal_acc"   # "acc" | "bal_acc" | "youden"
target_tpr = 0.7                # 针对类1(NC)的TPR目标；不需要就设为 None


# ----------------- 训练一个epoch -----------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
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

        total_loss += loss.detach().item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ----------------- Subject-level评估工具 -----------------
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
        return max(m, key=len)  # 最长数字串
    stem = os.path.splitext(os.path.basename(p))[0]
    return stem.split("_")[0]

def _pick_threshold_with_target_tpr(sid_true, sid_probs, target_tpr=None, metric="bal_acc"):
    thrs = np.linspace(0.0, 1.0, 2001)

    def _tpr_tnr(pred, true):
        tp = ((pred == 1) & (true == 1)).sum()
        fn = ((pred == 0) & (true == 1)).sum()
        tn = ((pred == 0) & (true == 0)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        tpr = tp / max(tp + fn, 1)  # 类1(NC)的TPR
        tnr = tn / max(tn + fp, 1)  # 类0(AD)的TNR
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
def evaluate_subject_level_split(model, loader, criterion, device,
                                 threshold=None, search_thresh=False,
                                 metric_for_search="acc"):
    """
    返回: epoch_loss, acc_overall, auc_overall, best_thr, per_class
    per_class: {AD_n, NC_n, AD_acc, NC_acc, TP, TN, FP, FN, recall_AD, specificity_NC}
    说明：labels中约定 0=AD, 1=NC；计算概率用 softmax(logits)[:,1] 表示NC的概率。
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

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # NC 概率
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs]); ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

    prob_by_sid = defaultdict(list); label_by_sid = {}
    for prob, y, p in zip(y_prob_img, y_true_img, paths_this_epoch):
        sid = _extract_subject_id(p)
        prob_by_sid[sid].append(prob)
        label_by_sid.setdefault(sid, []).append(int(y))

    sid_list = sorted(prob_by_sid.keys())
    sid_probs = np.array([np.mean(prob_by_sid[s]) for s in sid_list], dtype=np.float32)
    sid_true  = np.array([int(round(np.mean(label_by_sid[s]))) for s in sid_list], dtype=np.int64)

    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        _metric = metric_for_search if metric_for_search is not None else globals().get("metric_for_search", "acc")
        _target_tpr = globals().get("target_tpr", None)
        best_thr = _pick_threshold_with_target_tpr(sid_true, sid_probs, target_tpr=_target_tpr, metric=_metric)

    sid_pred = (sid_probs > best_thr).astype(int)  # 1=NC
    acc_overall = accuracy_score(sid_true, sid_pred)
    try:
        auc_overall = roc_auc_score(sid_true, sid_probs)
    except Exception:
        auc_overall = 0.0

    mask_AD = (sid_true == 0); mask_NC = (sid_true == 1)
    AD_n = int(mask_AD.sum()); NC_n = int(mask_NC.sum())
    AD_acc = float(((sid_pred == 0) & mask_AD).sum() / AD_n) if AD_n > 0 else float("nan")
    NC_acc = float(((sid_pred == 1) & mask_NC).sum() / NC_n) if NC_n > 0 else float("nan")

    TP = int(((sid_pred == 0) & (sid_true == 0)).sum())
    TN = int(((sid_pred == 1) & (sid_true == 1)).sum())
    FP = int(((sid_pred == 0) & (sid_true == 1)).sum())
    FN = int(((sid_pred == 1) & (sid_true == 0)).sum())

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

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # NC 概率
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs]); ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

    from statistics import median
    prob_by_sid = defaultdict(list); label_by_sid = {}
    for prob, y, p in zip(y_prob_img, y_true_img, paths_this_epoch):
        sid = _extract_subject_id(p)
        prob_by_sid[sid].append(prob)
        label_by_sid.setdefault(sid, []).append(int(y))

    sid_list = sorted(prob_by_sid.keys())
    sid_probs = np.array([median(prob_by_sid[s]) for s in sid_list], dtype=np.float32)
    sid_true  = np.array([int(round(np.mean(label_by_sid[s]))) for s in sid_list], dtype=np.int64)

    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        best_thr = _pick_threshold_with_target_tpr(
            sid_true, sid_probs, target_tpr=target_tpr, metric=metric_for_search
        )

    acc = accuracy_score(sid_true, (sid_probs > best_thr).astype(int))
    try:
        auc = roc_auc_score(sid_true, sid_probs)
    except Exception:
        auc = 0.0
    return float(epoch_loss), float(acc), float(auc), float(best_thr)


# ----------------- 绘图 -----------------
def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"],   label="val_acc")
    if "test_acc" in history and len(history["test_acc"]) == len(history["val_acc"]):
        plt.plot(history["test_acc"],  label="test_acc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png")); plt.close()

    plt.figure()
    plt.plot(history["val_auc"],  label="val_auc")
    if "test_auc" in history and len(history["test_auc"]) == len(history["val_auc"]):
        plt.plot(history["test_auc"], label="test_auc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("AUC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png")); plt.close()

    if "lr" in history and len(history["lr"]) > 0:
        plt.figure()
        plt.plot(history["lr"], label="lr")
        plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lr_curve.png")); plt.close()

    if "best_thr" in history and len(history["best_thr"]) > 0:
        plt.figure()
        plt.plot(history["best_thr"], label="best_thr (val subject-level)")
        plt.xlabel("epoch"); plt.ylabel("threshold"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_curve.png")); plt.close()


# ----------------- 其他工具 -----------------
def _collect_labels(ds):
    if hasattr(ds, 'targets'):
        return list(map(int, ds.targets))
    if hasattr(ds, 'samples'):
        return [int(cls) for _, cls in ds.samples]
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

    # Data
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
        cnt = Counter(labs); total = sum(cnt.values())
        w0 = total / (2.0 * max(cnt.get(0, 1), 1))
        w1 = total / (2.0 * max(cnt.get(1, 1), 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Class weights -> class0: {w0:.4f}, class1: {w1:.4f}")

    # Model
    model = build_model(in_channels=args.in_channels, height=args.img_size, width=args.img_size).to(device)

    # Loss & Optim
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

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

    # Warmup + Cosine LR
    base_lr = args.lr; min_lr = 1e-6; warmup_epochs = 3
    def compute_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        T = max(1, args.epochs - warmup_epochs)
        import math
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))

    scaler = GradScaler('cuda')

    best_val_auc = -1.0
    best_thr = 0.5
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [],
               "val_auc": [], "test_acc": [], "test_auc": [],
               "lr": [], "best_thr": []}

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        cur_lr = compute_lr(epoch); set_lr(optimizer, cur_lr)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        val_loss, val_acc, val_auc, best_thr_epoch = evaluate_subject_level(
            model, val_loader, criterion, device, search_thresh=True
        )

        _, test_acc, test_auc, _, test_split = evaluate_subject_level_split(
            model, test_loader, criterion, device,
            threshold=best_thr_epoch, search_thresh=False
        )

        # 记录
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["test_acc"].append(test_acc);  history["test_auc"].append(test_auc)
        history["lr"].append(cur_lr);          history["best_thr"].append(best_thr_epoch)

        # 打印
        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, BestThr: {best_thr_epoch:.3f}")
        print(f"Test  Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"Per-class Test Acc -> AD: {test_split['AD_acc']:.4f} (n={test_split['AD_n']}), "
              f"NC: {test_split['NC_acc']:.4f} (n={test_split['NC_n']})")
        print(f"Confusion (AD positive): TP={test_split['TP']}  FP={test_split['FP']}  "
              f"TN={test_split['TN']}  FN={test_split['FN']}")
        print(f"Recall_AD={test_split['recall_AD']:.4f}, Specificity_NC={test_split['specificity_NC']:.4f}")
        print(f"LR after epoch {epoch+1}: {cur_lr:.6f}")

        # 保存: 以 val_auc 为准
        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_thr = best_thr_epoch
            torch.save(
                {"model": model.state_dict(),
                 "args": vars(args),
                 "class_names": class_names,
                 "best_thr": best_thr},
                os.path.join(args.outdir, "best_model.pt")
            )
            print("✅ Saved best model (by subject-level AUC)")

    # -------- Final test with the best checkpoint --------
    ck = torch.load(os.path.join(args.outdir, "best_model.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    best_thr_final = float(ck.get("best_thr", best_thr))

    test_loss, test_acc, test_auc, _ = evaluate_subject_level(
        model, test_loader, criterion, device,
        threshold=best_thr_final, search_thresh=False
    )
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f} (thr={best_thr_final:.3f})")

    plot_curves(history, args.outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ADNI/AD_NC")
    ap.add_argument("--outdir", type=str, default="runs/adni_gfnet_minimal")
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
