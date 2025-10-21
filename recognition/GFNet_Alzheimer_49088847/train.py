import os
import torch
import torch.nn as nn
import argparse
from modules import build_model
from dataset import get_loaders 
from dataset import set_seed
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_prob, total_loss = [], [], 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, 1)[:,1].cpu().numpy()
        y_prob.extend(probs)
        y_true.extend(labels.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)
    acc = accuracy_score(y_true, (torch.tensor(y_prob)>0.5).numpy())
    auc = roc_auc_score(y_true, y_prob)
    return total_loss / len(loader.dataset), acc, auc

def plot_curves(history, outdir, lrs=None):
   
    os.makedirs(outdir, exist_ok=True)

    # ---- 1️⃣ Loss ----
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # ---- 2️⃣ Accuracy ----
    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    if "test_acc" in history:
        plt.plot(history["test_acc"], label="Test Acc", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png"))
    plt.close()

    # ---- 3️⃣ AUC ----
    plt.figure()
    plt.plot(history["val_auc"], label="Val AUC")
    if "test_auc" in history:
        plt.plot(history["test_auc"], label="Test AUC", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("AUC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png"))
    plt.close()

    # ---- 4️⃣ Learning Rate ----
    if lrs is not None:
        plt.figure()
        plt.plot(lrs, label="Learning Rate", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.legend()
        plt.title("Learning Rate Schedule")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lr_curve.png"))
        plt.close()

    # ---- 5️⃣ 保存历史数据 ----
    import json
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)
    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        gray=(args.in_channels == 1)
    )

 
    model = build_model(in_channels=args.in_channels, height=args.img_size, width=args.img_size)


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

  
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")


        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")


        test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")


    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pt"))
            print("✅ Saved best model")


    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ADNI/AD_NC")
    ap.add_argument("--outdir", type=str, default="runs/adni_resnet")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--in_channels", type=int, default=1, help="ADNI 多为灰度：1")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
