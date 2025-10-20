import os, argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from modules import build_model



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
    
@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    model_args = ckpt.get("args", {})
    class_names = ckpt.get("class_names", ["CN","AD"])  # 以训练时保存为准

    model = build_model(in_channels=model_args.get("in_channels",1),
                        backbone=model_args.get("backbone","resnet18"),
                        pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    tf = build_transform(img_size=model_args.get("img_size",224),
                         gray=(model_args.get("in_channels",1)==1))

    imgs = []
    for p in args.images:
        img = Image.open(p).convert("RGB")
