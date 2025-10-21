import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modules import build_model


def build_transform(img_size=224, gray=True):

    t = [transforms.Resize((img_size, img_size))]
    if gray:
        t = [transforms.Grayscale(num_output_channels=1)] + t
    t += [transforms.ToTensor()]
    if gray:
        t += [transforms.Normalize([0.5], [0.5])]
    else:
        t += [transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")


    ckpt = torch.load(args.weights, map_location=device)
    model_args = ckpt.get("args", {})
    class_names = ckpt.get("class_names", ["NC", "AD"])

    model = build_model(
        in_channels=model_args.get("in_channels", 1),
        height=model_args.get("img_size", 224),
        width=model_args.get("img_size", 224)
    )

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print("✅ Model loaded successfully.")


    gray = model_args.get("in_channels", 1) == 1
    tf = build_transform(img_size=model_args.get("img_size", 224), gray=gray)


    for p in args.images:
        if not os.path.exists(p):
            print(f"⚠️ Image not found: {p}")
            continue

        img = Image.open(p)
        img = img.convert("L") if gray else img.convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

        logits = model(x)
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = prob.argmax()

        print(f"\n🖼 {os.path.basename(p)}")
        for i, cls in enumerate(class_names):
            print(f"  {cls}: {prob[i]:.3f}")
        print(f"➡️  Predicted: **{class_names[pred_idx]}** (Confidence: {prob[pred_idx]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--images", nargs="+", required=True, help="List of image paths to predict")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = ap.parse_args()
    main(args)
