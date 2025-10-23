import os, json, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import re

def _extract_subject_id(p):
    """
    Extract a subject ID string from a file path.

    Priority 1: any run of ≥5 digits found anywhere in the path (common for ADNI IDs).
    Priority 2: if none found, fall back to the file stem's first token split by underscore.

    Args:
        p (str): File path.

    Returns:
        str: Best-effort subject identifier derived from the path.
    """
    m = re.findall(r"\d{5,}", p.replace("\\", "/"))
    if m:
        # If multiple numeric chunks exist, take the longest one.
        return max(m, key=len)
    stem = os.path.splitext(os.path.basename(p))[0]
    return stem.split("_")[0]

def set_seed(seed: int = 42):
    """
    Set Python, PyTorch (CPU) and CUDA RNG seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomGamma:
    """
    Random gamma correction as a torchvision-style callable transform.

    - With probability p, raise pixel values to a random gamma in [lo, hi].
    - Works by converting PIL -> tensor -> power -> PIL to avoid precision loss.
    """
    def __init__(self, gamma_range=(0.95, 1.05), p=0.4):
        self.lo, self.hi = gamma_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            g = random.uniform(self.lo, self.hi)
            t = transforms.functional.to_tensor(img)
            # Clamp to avoid taking power of 0; keep values in [0,1]
            t = torch.clamp(t, 1e-6, 1.0) ** g
            return transforms.functional.to_pil_image(t)
        return img


class AddGaussianNoise:
    """
    Additive Gaussian noise as a torchvision-style callable transform.

    - With probability p, add N(0, std^2) noise to the image tensor.
    - Clamps the result to [0, 1], then converts back to PIL Image.
    """
    def __init__(self, std=0.005, p=0.2):
        self.std = std
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            t = transforms.functional.to_tensor(img)
            t = torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)
            return transforms.functional.to_pil_image(t)
        return img


def get_transforms(img_size=224, gray=True, aug=True):
    """
    Build torchvision transforms for training/eval.

    Args:
        img_size (int): Resize target (square).
        gray (bool): If True, convert to single-channel and use grayscale normalization.
        aug (bool): If True, include light geometric/intensity/noise augmentation.

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.
    """
    if aug:
        t = [
            transforms.Resize((img_size, img_size)),
            # Small rotations to simulate acquisition variation.
            transforms.RandomRotation(10),
            # Tiny translations/scales; degrees=0 -> no extra rotation here.
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02)),
            RandomGamma((0.95, 1.05), p=0.4),
            AddGaussianNoise(std=0.005, p=0.2),
        ]
    else:
        t = [transforms.Resize((img_size, img_size))]

    if gray:
        # Force grayscale and keep a single output channel.
        t = [transforms.Grayscale(num_output_channels=1)] + t

    # Convert PIL to tensor and normalize.
    t += [transforms.ToTensor()]
    if gray:
        # Standardize single-channel to mean=0.5, std=0.5
        t += [transforms.Normalize([0.5], [0.5])]
    else:
        # Standard ImageNet normalization for RGB images.
        t += [transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


def split_by_subject(meta_json_path, val_ratio=0.1, seed=42):
    """
    Split subjects into train/val sets using a subject-level metadata JSON.

    The metadata JSON is expected to map subject_id -> info dict, with at least:
        info["label"]: numeric label where 0 means AD and 1 means NC (as assumed here).

    Args:
        meta_json_path (str): Path to metadata JSON.
        val_ratio (float): Fraction of subjects to assign to validation.
        seed (int): Random seed for deterministic split.

    Returns:
        (set[str], set[str]): Train subject IDs, Val subject IDs.
    """
    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    all_subjects = list(meta.keys())
    train_subj, val_subj = train_test_split(all_subjects, test_size=val_ratio, random_state=seed)

    # Simple label statistics (AD vs NC) at the SUBJECT level
    label_count = {"AD": 0, "NC": 0}
    val_label_count = {"AD": 0, "NC": 0}
    for sid, info in meta.items():
        # Assumption: label==0 -> AD, label==1 -> NC
        lab = "AD" if info["label"] == 0 else "NC"
        if sid in train_subj:
            label_count[lab] += 1
        else:
            val_label_count[lab] += 1
    print(f"✅ Train subjects: {len(train_subj)}, Val subjects: {len(val_subj)}")
    print(f"Train label stats: {label_count}")
    print(f"Val   label stats: {val_label_count}")

    return set(train_subj), set(val_subj)


def get_loaders(data_root="ADNI/AD_NC",
                meta_json="ADNI/meta_data_with_label.json",
                img_size=224, batch_size=32,
                num_workers=8, gray=True, seed=42):
    """
    Create PyTorch DataLoaders with SUBJECT-LEVEL train/val split.

    Directory layout (example):
        data_root/
          train/AD/*images*
          train/NC/*images*
          test/AD/*images*
          test/NC/*images*

    Returns:
        (DataLoader, DataLoader, DataLoader, list[str]):
            train_loader, val_loader, test_loader, class_names
    """
    set_seed(seed)

    # Expand to absolute path if a relative meta_json path was provided.
    if not os.path.isabs(meta_json):
        meta_json = os.path.join(os.getcwd(), meta_json)

    # Subject-level split (e.g., 80/20)
    train_subj, val_subj = split_by_subject(meta_json, val_ratio=0.2, seed=seed)

    # Build transforms
    tf_train = get_transforms(img_size, gray=gray, aug=True)
    tf_eval  = get_transforms(img_size, gray=gray, aug=False)

    # Load the same physical images from 'train', but we'll index them differently
    # based on which subject they belong to.
    full_train = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_train)
    # Use eval transforms for the validation VIEW, still reading from 'train' directory.
    full_val   = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_eval)

    def get_subject_id_from_path(p):
        # Thin wrapper in case we need to swap subject-id parsing later.
        return _extract_subject_id(p)

    # Build index lists for subset sampling
    train_indices = []
    val_indices   = []
    unk_train = unk_val = 0  # Counters for files whose subject IDs aren't found in split sets

    # Scan all training samples; keep those whose subject ID is in train_subj
    for i, (path, _) in enumerate(full_train.samples):
        sid = get_subject_id_from_path(path)
        if sid in train_subj:
            train_indices.append(i)
        elif sid in val_subj:
            # Present in val set -> skip here, will be picked up in the val view below
            pass
        else:
            # Subject in metadata not matched, or missing from both sets
            unk_train += 1

    # Scan again (eval transform view) to capture validation indices only
    for i, (path, _) in enumerate(full_val.samples):
        sid = get_subject_id_from_path(path)
        if sid in val_subj:
            val_indices.append(i)
        elif sid in train_subj:
            # Present in train set -> skip here
            pass
        else:
            unk_val += 1

    if unk_train or unk_val:
        print(f"⚠️  Samples with unknown subject (ignored): train_view={unk_train}, val_view={unk_val}")

    # Create subset datasets for train/val without duplicating files on disk
    train_ds = torch.utils.data.Subset(full_train, train_indices)
    val_ds   = torch.utils.data.Subset(full_val,   val_indices)

    # Test set lives under its own directory and uses eval transforms
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=tf_eval)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Class names inferred from folder names under data_root/train (e.g., ["AD", "NC"])
    class_names = full_train.classes
    return train_loader, val_loader, test_loader, class_names