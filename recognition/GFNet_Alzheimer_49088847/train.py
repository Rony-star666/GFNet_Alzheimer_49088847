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

metric_for_search = "bal_acc"   # Options: "acc" | "bal_acc" | "youden"
target_tpr = 0.7               # e.g., 0.75; set to None to disable

# ----------------- Temperature Scaling -----------------
class _TempScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))  # start with T=1
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
    After each epoch, adaptively make small regularization adjustments based on
    the train/val performance gap.
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

        self.best_val = -1.0       # ✅ use a member variable to track best val metric
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
        # "improve" is decided by the caller
        if improve:
            self.best_val = max(self.best_val, val_auc)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        gap = float(tr_acc - val_acc)
        wd = self._get_weight_decay()
        ls = self._get_label_smoothing()
        dp = self._get_dropout()

        # Overfitting: large gap and no improvement -> increase regularization
        if gap > self.gap_hi and self.bad_epochs >= 1:
            target = wd * 1.5 if wd > 0 else self.wd_bounds[0]
            self._set_weight_decay(target)
            self._set_label_smoothing(ls + 0.01)
            self._set_dropout(dp + 0.02)
            if self.ema is not None and isinstance(self.ema, dict):
                self.ema["decay"] = min(0.9999, self.ema["decay"] + 0.0005)

        # Underfitting: small gap and low val accuracy -> decrease regularization
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
        raise RuntimeError("The path cannot be restored from the current dataset; Please ensure that the DataLoader shuffle of val/test =False.")
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
            # Fallback: maximize the selected metric over all thresholds
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
    temp_layer=None,
    metric_for_search="acc"  # Can be overridden by the caller; otherwise use this default or the global value
):
    """
    Similar to evaluate_subject_level, but additionally returns per-class (AD=0, NC=1)
    accuracy and a confusion matrix (treat AD as the "positive" class).

    Returns: epoch_loss, acc_overall, auc_overall, best_thr, per_class
    per_class = {
        "AD_n": int, "NC_n": int,
        "AD_acc": float, "NC_acc": float,
        "TP": int, "TN": int, "FP": int, "FN": int,
        "recall_AD": float,     # sensitivity for AD
        "specificity_NC": float # specificity for NC
    }
    """
    model.eval()
    running_loss, y_true_img, y_prob_img = 0.0, [], []

    # Paths in the same order as the DataLoader (requires shuffle=False for val/test)
    paths_all = _ordered_paths_from_loader(loader)
    ptr = 0
    paths_this_epoch = []

    for imgs, labels in loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        logits = _with_temp_forward(model, temp_layer, imgs)
        loss = criterion(logits, labels)
        running_loss += loss.detach().item() * bs

        # Probability definition: use probability of class 1 (NC);
        # therefore (prob > thr) -> predict NC
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_prob_img.extend(probs.tolist())
        y_true_img.extend(labels.detach().cpu().numpy().tolist())

        paths_this_epoch.extend(paths_all[ptr:ptr+bs])
        ptr += bs

    epoch_loss = running_loss / max(len(loader.dataset), 1)

    # ---- Aggregate to subject level (mean; could switch to median) ----
    prob_by_sid = defaultdict(list)
    label_by_sid = {}
    for prob, y, p in zip(y_prob_img, y_true_img, paths_this_epoch):
        sid = _extract_subject_id(p)
        prob_by_sid[sid].append(prob)
        label_by_sid.setdefault(sid, []).append(int(y))

    sid_list = sorted(prob_by_sid.keys())
    sid_probs = np.array([np.mean(prob_by_sid[s]) for s in sid_list], dtype=np.float32)
    sid_true  = np.array([int(round(np.mean(label_by_sid[s]))) for s in sid_list], dtype=np.int64)

    # ---- Pick threshold (at subject level) ----
    best_thr = 0.5 if threshold is None else float(threshold)
    if search_thresh:
        # If the caller doesn't provide metric_for_search, fall back to the global variable
        _metric = metric_for_search if metric_for_search is not None else globals().get("metric_for_search", "acc")
        _target_tpr = globals().get("target_tpr", None)
        best_thr = _pick_threshold_with_target_tpr(
            sid_true, sid_probs,
            target_tpr=_target_tpr,
            metric=_metric
        )

    # ---- Apply threshold and compute metrics ----
    # Note: prob > thr => predict class "1=NC"; therefore predicting AD is (pred==0)
    sid_pred = (sid_probs > best_thr).astype(int)

    acc_overall = accuracy_score(sid_true, sid_pred)
    try:
        auc_overall = roc_auc_score(sid_true, sid_probs)
    except Exception:
        auc_overall = 0.0

    # Per-class stats (treat AD=0 as the "positive" class to report recall conveniently)
    mask_AD = (sid_true == 0)
    mask_NC = (sid_true == 1)
    AD_n = int(mask_AD.sum())
    NC_n = int(mask_NC.sum())

    # AD_acc: among true AD, proportion predicted as AD, i.e., (pred==0) within true AD subset
    AD_acc = float(((sid_pred == 0) & mask_AD).sum() / AD_n) if AD_n > 0 else float("nan")
    # NC_acc: among true NC, proportion predicted as NC, i.e., (pred==1) within true NC subset
    NC_acc = float(((sid_pred == 1) & mask_NC).sum() / NC_n) if NC_n > 0 else float("nan")

    # Confusion matrix elements (with AD as the "positive" class)
    TP = int(((sid_pred == 0) & (sid_true == 0)).sum())  # true AD and predicted AD
    TN = int(((sid_pred == 1) & (sid_true == 1)).sum())  # true NC and predicted NC
    FP = int(((sid_pred == 0) & (sid_true == 1)).sum())  # true NC but predicted AD
    FN = int(((sid_pred == 1) & (sid_true == 0)).sum())  # true AD but predicted NC

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
                           threshold=None, search_thresh=False, temp_layer=None):
    model.eval()
    running_loss, y_true_img, y_prob_img = 0.0, [], []

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
            target_tpr=target_tpr,      # ← uses global setting
            metric=metric_for_search    # ← uses global setting
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

    # 6) AutoTuner regularization trajectories
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
    # Compatible with ImageFolder / custom datasets
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

    # Load DataLoaders
    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        gray=(args.in_channels == 1)
    )

    # Class weights for imbalance handling (optional)
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

    # Warmup + Cosine (manually adjust LR to avoid conflicting with EMA updates)
    base_lr = args.lr
    min_lr = 1e-6
    warmup_epochs = 3

    def compute_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        # cosine schedule from base_lr -> min_lr
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
    best_val_auc = -1.0    # ✅ initialize AUC baseline
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

        # ---- Temperature scaling (fit on validation set; never on test) ----
        if epoch == 0 or (epoch % 5 == 0):
            temp_layer = fit_temperature(ema_model, val_loader, device)
            for p in temp_layer.parameters():
                p.requires_grad_(False)

        # ---- Validation (subject-level + temperature) ----
        val_loss, val_acc, val_auc, best_thr_epoch = evaluate_subject_level(
            ema_model, val_loader, criterion, device, search_thresh=True, temp_layer=temp_layer
        )

        # ---- Test (subject-level + temperature, using the best Val threshold) ----
        test_loss, test_acc, test_auc, _, test_split = evaluate_subject_level_split(
            ema_model, test_loader, criterion, device,
            threshold=best_thr_epoch, search_thresh=False, temp_layer=temp_layer
        )

        # Logging
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

        # AutoTuner (use AUC to decide improvement)
        eps = 1e-6
        improve = (val_auc > best_val_auc + eps)   # ✅ decide improvement by AUC
        info = tuner.step(tr_acc, val_acc, val_auc, improve=improve)

        history["wd"].append(info["wd"])
        history["ls"].append(info["ls"])
        history["dp"].append(info["dp"])
        print(f"[AutoTuner] wd={info['wd']:.2e} ls={info['ls']:.3f} dp={info['dp']:.2f} "
              f"gap={info['gap']:.3f} bad_epochs={info['bad_epochs']} (ema_decay={ema_decay_ref['decay']:.5f})")

        # Save: select by best val_auc
        if val_auc > best_val_auc + eps:
            best_val_auc = val_auc              # ✅ update the "best AUC"
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

    # Fit temperature on the validation set again for consistency
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
