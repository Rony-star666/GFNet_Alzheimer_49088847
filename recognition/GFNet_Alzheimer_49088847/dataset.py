import os, json, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomGamma:
    def __init__(self, gamma_range=(0.95, 1.05), p=0.4):
        self.lo, self.hi = gamma_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            g = random.uniform(self.lo, self.hi)
            t = transforms.functional.to_tensor(img)
            t = torch.clamp(t, 1e-6, 1.0) ** g
            return transforms.functional.to_pil_image(t)
        return img


class AddGaussianNoise:
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
    if aug:
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02)),
            RandomGamma((0.95, 1.05), p=0.4),
            AddGaussianNoise(std=0.005, p=0.2),
        ]
    else:
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


def split_by_subject(meta_json_path, val_ratio=0.1, seed=42):
    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    all_subjects = list(meta.keys())
    train_subj, val_subj = train_test_split(all_subjects, test_size=val_ratio, random_state=seed)


    label_count = {"AD": 0, "NC": 0}
    val_label_count = {"AD": 0, "NC": 0}
    for sid, info in meta.items():
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
                num_workers=4, gray=True, seed=42):

    set_seed(seed)

    if not os.path.isabs(meta_json):
        meta_json = os.path.join(os.getcwd(), meta_json)
    train_subj, val_subj = split_by_subject(meta_json, val_ratio=0.1, seed=seed)

    tf_train = get_transforms(img_size, gray=gray, aug=True)
    tf_eval  = get_transforms(img_size, gray=gray, aug=False)


    full_train = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_train)


    def get_subject_id_from_path(p):
        for sid in train_subj | val_subj:
            if sid in p:
                return sid
        return None


    train_indices, val_indices = [], []
    for i, (path, _) in enumerate(full_train.samples):
        sid = get_subject_id_from_path(path)
        if sid is None:
            train_indices.append(i)
        elif sid in val_subj:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_ds = torch.utils.data.Subset(full_train, train_indices)
    val_ds   = torch.utils.data.Subset(
        datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_eval),
        val_indices
    )


    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=tf_eval)

  
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = full_train.classes
    return train_loader, val_loader, test_loader, class_names
