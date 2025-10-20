import torch
import argparse

def main(args):
    print("Prediction script initialized")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()
    main(args)