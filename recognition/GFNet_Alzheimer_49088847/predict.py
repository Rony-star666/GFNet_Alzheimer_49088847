import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modules import build_model

"""
Simple CLI for running inference on one or more images using a saved
checkpoint (.pt). The script restores model hyperparameters from the
checkpoint (e.g., in_channels/img_size) to build the same architecture
and preprocessing pipeline used during training.
"""


def build_transform(img_size=224, gray=True):
    """Return a torchvision transform that matches training preprocessing.

    Args:
        img_size (int): Target square size (H=W) for resizing.
        gray (bool): If True, convert to single-channel grayscale
            and normalize with mean=std=0.5. If False, keep RGB and
            use ImageNet mean/std.
    """
    # Always resize to (img_size, img_size)
    t = [transforms.Resize((img_size, img_size))]

    # Insert grayscale conversion at the beginning if needed
    if gray:
        t = [transforms.Grayscale(num_output_channels=1)] + t

    # Convert PIL image to tensor in [0,1]
    t += [transforms.ToTensor()]

    # Per-channel normalization (must match training)
    if gray:
        # For 1-channel input: mean=0.5, std=0.5
        t += [transforms.Normalize([0.5], [0.5])]
    else:
        # For 3-channel RGB input: ImageNet statistics
        t += [transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


@torch.no_grad()
def main(args):
    """Run inference for the provided image paths.

    Notes:
        * Device is CUDA if available unless --cpu is set.
        * The model architecture and preprocessing are derived
          from the saved checkpoint's 'args'.
        * Class names are loaded from the checkpoint if available.
    """
    # Select device: CUDA preferred; override with --cpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint (supports state dict-only or dict with keys)
    ckpt = torch.load(args.weights, map_location=device)

    # Retrieve saved training args and class names (fallbacks provided)
    model_args = ckpt.get("args", {})
    class_names = ckpt.get("class_names", ["NC", "AD"])  # expected order must match training

    # Rebuild the model using saved hyperparameters so shapes match
    model = build_model(
        in_channels=model_args.get("in_channels", 1),
        height=model_args.get("img_size", 224),
        width=model_args.get("img_size", 224)
    )

    # Load weights; accept either {'model': state_dict, ...} or pure state_dict
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print("✅ Model loaded successfully.")

    # Build the preprocessing transform to match training
    gray = model_args.get("in_channels", 1) == 1
    tf = build_transform(img_size=model_args.get("img_size", 224), gray=gray)

    # Iterate through images given on the command line
    for p in args.images:
        if not os.path.exists(p):
            # Skip missing files but continue with the rest
            print(f"⚠️ Image not found: {p}")
            continue

        # Open image with PIL and ensure correct mode (L for 1ch, RGB for 3ch)
        img = Image.open(p)
        img = img.convert("L") if gray else img.convert("RGB")

        # Apply transforms and add batch dimension: (B=1, C, H, W)
        x = tf(img).unsqueeze(0).to(device)

        # Forward pass
        logits = model(x)

        # Convert logits to probabilities via softmax
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = prob.argmax()

        # Pretty print per-class probabilities and the top prediction
        print(f"\n🖼 {os.path.basename(p)}")
        for i, cls in enumerate(class_names):
            print(f"  {cls}: {prob[i]:.3f}")
        print(f"➡️  Predicted: **{class_names[pred_idx]}** (Confidence: {prob[pred_idx]:.3f})")


if __name__ == "__main__":
    # CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--images", nargs="+", required=True, help="List of image paths to predict")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = ap.parse_args()

    # Run inference
    main(args)
