# train.py —— no-EMA & no-Temperature-Scaling
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

metric_for_search = "bal_acc"   # "acc" | "bal_acc" | "youden"
target_tpr = 0.7                # 目标召回(针对 NC=1 的 TPR)，不需要就置为 None


# ----------------- AutoTuner -----------------
class AutoTuner:
    def __init__(self, model, optimizer, criterion, ema=None,
                 gap_hi=0.08, gap_lo=0.02, patience=3,
                 wd_bounds=(1e-4, 5e-2), ls_bounds=(0.0, 0.10),
                 dp_bounds=(0.0, 0.4)):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.gap_hi = gap_hi
        self.gap_lo = gap_lo
        self.patience = patience
        self.wd_bounds = wd_bounds
        self.ls_bounds = ls_bounds
        self.dp_bounds = dp_bounds
        self.best_val = -1.0
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
        ps = [m.p for m in self.model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
        return sum(ps)/len(ps) if ps else 0.0

    def step(self, tr_acc, val_acc, val_auc, improve, tf_train=None):
        if improve:
            self.best_val = max(self.best_val, val_auc)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        gap = float(tr_acc - val_acc)
        wd = self._get_weight_decay()
        ls = self._get_label_smoothing()
        dp = self._get_dropout()

        if gap > self.gap_hi and self.bad_epochs >= 1:
            target = wd * 1.5 if wd > 0 else self.wd_bounds[0]
            self._set_weight_decay(target)
            self._set_label_smoothing(ls + 0.01)
            self._set_dropout(dp + 0.02)
        elif gap < self.gap_lo and val_acc < 0.65 and self.bad_epochs >= 2:
            self._set_weight_decay(wd * 0.7)
            self._set_label_smoothing(max(self.ls_bounds[0], ls - 0.01))
            self._set_dropout(max(self.dp_bounds[0], dp - 0.02))

        return {"wd": self._get_weight_decay(),
                "ls": self._get_label_smoothing(),
                "dp": self._get_dropout(),
                "bad_epochs": self.bad_epochs,
                "gap": gap}


# ----------------- Train one epoch -----------------
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
        return max(m, key=len)
    stem = os.path.splitext(os.path.basename(p))[0]
    return stem.split("_")[0]

def _pick_threshold_with_target_tpr(sid_true, sid_probs, target_tpr=None, metric="bal_acc"):
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
        else:
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

    sid_pred = (sid_probs > best_thr).astype(int)
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
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
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


# ----------------- Plot utils -----------------
def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(); plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()

    plt.figure(); plt.plot(history["train_acc"], label="train_acc"); plt.plot(history["val_acc"], label="val_acc")
    if "test_acc" in history and len(history["test_acc"]) == len(history["val_acc"]):
        plt.plot(history["test_acc"], label="test_acc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png")); plt.close()

    plt.figure(); plt.plot(history["val_auc"], label="val_auc")
    if "test_auc" in history and len(history["test_auc"]) == len(history["val_auc"]):
        plt.plot(history["test_auc"], label="test_auc", alpha=0.8)
    plt.xlabel("epoch"); plt.ylabel("AUC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png")); plt.close()

    if "lr" in history and len(history["lr"]) > 0:
        plt.figure(); plt.plot(history["lr"], label="lr")
        plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lr_curve.png")); plt.close()

    if "best_thr" in history and len(history["best_thr"]) > 0:
        plt.figure(); plt.plot(history["best_thr"], label="best_thr (val subject-level)")
        plt.xlabel("epoch"); plt.ylabel("threshold"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_curve.png")); plt.close()

    if "wd" in history and len(history["wd"]) > 0:
        plt.figure(); plt.plot(history["wd"], label="weight_decay")
        plt.xlabel("epoch"); plt.ylabel("weight_decay"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "wd_curve.png")); plt.close()
    if "ls" in history and len(history["ls"]) > 0:
        plt.figure(); plt.plot(history["ls"], label="label_smoothing")
        plt.xlabel("epoch"); plt.ylabel("label_smoothing"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "label_smoothing_curve.png")); plt.close()
    if "dp" in history and len(history["dp"]) > 0:
        plt.figure(); plt.plot(history["dp"], label="dropout_p")
        plt.xlabel("epoch"); plt.ylabel("dropout"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "dropout_curve.png")); plt.close()


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

    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.workers,
        gray=(args.in_channels == 1)
    )

    labs = _collect_labels(train_loader.dataset)
    class_weights = None
    if labs:
        cnt = Counter(labs); total = sum(cnt.values())
        w0 = total / (2.0 * max(cnt.get(0, 1), 1))
        w1 = total / (2.0 * max(cnt.get(1, 1), 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Class weights -> class0: {w0:.4f}, class1: {w1:.4f}")

    model = build_model(in_channels=args.in_channels, height=args.img_size, width=args.img_size).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr
    )

    base_lr = args.lr; min_lr = 1e-6; warmup_epochs = 3
    def compute_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        T = max(1, args.epochs - warmup_epochs)
        import math
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))

    tuner = AutoTuner(model=model, optimizer=optimizer, criterion=criterion,
                      gap_hi=0.08, gap_lo=0.02, patience=3,
                      wd_bounds=(1e-4, 5e-2), ls_bounds=(0.0, 0.10), dp_bounds=(0.0, 0.4))
    scaler = GradScaler('cuda')

    best_val_auc = -1.0
    best_thr = 0.5
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "val_auc": [], "test_acc": [], "test_auc": [],
               "lr": [], "best_thr": [], "wd": [], "ls": [], "dp": []}

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        cur_lr = compute_lr(epoch); set_lr(optimizer, cur_lr)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        val_loss, val_acc, val_auc, best_thr_epoch = evaluate_subject_level(
            model, val_loader, criterion, device, search_thresh=True
        )
        test_loss, test_acc, test_auc, _, test_split = evaluate_subject_level_split(
            model, test_loader, criterion, device,
            threshold=best_thr_epoch, search_thresh=False
        )

        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["test_acc"].append(test_acc);  history["test_auc"].append(test_auc)
        history["lr"].append(cur_lr);          history["best_thr"].append(best_thr_epoch)

        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, BestThr: {best_thr_epoch:.3f}")
        print(f"Test  Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"Per-class Test Acc -> AD: {test_split['AD_acc']:.4f} (n={test_split['AD_n']}), "
              f"NC: {test_split['NC_acc']:.4f} (n={test_split['NC_n']})")
        print(f"Confusion (AD positive): TP={test_split['TP']}  FP={test_split['FP']}  "
              f"TN={test_split['TN']}  FN={test_split['FN']}")
        print(f"Recall_AD={test_split['recall_AD']:.4f}, Specificity_NC={test_split['specificity_NC']:.4f}")
        print(f"LR after epoch {epoch+1}: {cur_lr:.6f}")

        eps = 1e-6
        improve = (val_auc > best_val_auc + eps)
        info = tuner.step(tr_acc, val_acc, val_auc, improve=improve)
        history["wd"].append(info["wd"]); history["ls"].append(info["ls"]); history["dp"].append(info["dp"])
        print(f"[AutoTuner] wd={info['wd']:.2e} ls={info['ls']:.3f} dp={info['dp']:.2f} "
              f"gap={info['gap']:.3f} bad_epochs={info['bad_epochs']}")

        if improve:
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
    ap.add_argument("--outdir", type=str, default="runs/adni_gfnet_notemp")
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
