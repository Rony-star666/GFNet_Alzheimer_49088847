import torch
import argparse
from torchvision import transforms

def build_transform(img_size=224, gray=True):
    t = [transforms.Resize((img_size, img_size))]
    if gray:
        t = [transforms.Grayscale(num_output_channels=1)] + t
    t += [transforms.ToTensor()]
    if gray:
        t += [transforms.Normalize([0.5],[0.5])]
    else:
        t += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    return transforms.Compose(t)
    
def main(args):
    print("Prediction script initialized")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()
    main(args)