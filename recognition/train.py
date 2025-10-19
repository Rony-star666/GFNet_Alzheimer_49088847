import torch
import argparse

def main(args):
    print("Training script placeholder")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    main(args)
