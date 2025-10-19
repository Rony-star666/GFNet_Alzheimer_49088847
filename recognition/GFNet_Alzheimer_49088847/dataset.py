import os, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224, gray=True, aug=True):
    t = [transforms.Resize((img_size, img_size))]
    if aug:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
        ]
    t.append(transforms.ToTensor())
    if gray:
        t.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        return transforms.Compose([transforms.Grayscale(num_output_channels=1)] + t)
    else:
        t.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                      std=[0.229,0.224,0.225]))
        return transforms.Compose(t)

def get_loaders(data_root="ADNI/AD_NC", img_size=224, batch_size=32, num_workers=4, gray=True):
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    assert os.path.isdir(train_dir), f"Not found: {train_dir}"
    assert os.path.isdir(test_dir),  f"Not found: {test_dir}"

    tf_train = get_transforms(img_size, gray=gray, aug=True)
    tf_eval  = get_transforms(img_size, gray=gray, aug=False)

    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    test_ds  = datasets.ImageFolder(test_dir,  transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes
