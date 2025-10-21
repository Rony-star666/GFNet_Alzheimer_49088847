# dataset.py
import os, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224, gray=True, aug=True):
    t = []
    t.append(transforms.Resize((img_size, img_size)))
    if aug:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
        ]
    t.append(transforms.ToTensor())
    if gray:
     
        mean, std = 0.5, 0.5
        t.append(transforms.Normalize(mean=[mean], std=[std]))
    else:
 
        t.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(t)

def get_loaders(data_root="ADNI/AD_NC", img_size=224, batch_size=32, num_workers=4, gray=True):
   
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    tf_train = get_transforms(img_size, gray=gray, aug=True)
    tf_eval  = get_transforms(img_size, gray=gray, aug=False)

    prefix = [transforms.Grayscale(num_output_channels=1)] if gray else []
    tf_train = transforms.Compose(prefix + [tf_train])
    tf_eval  = transforms.Compose(prefix + [tf_eval])

    full_train = datasets.ImageFolder(train_dir, transform=tf_train)

    val_ratio = 0.1
    val_len = int(len(full_train) * val_ratio)
    train_len = len(full_train) - val_len
    train_ds, val_ds = torch.utils.data.random_split(full_train, [train_len, val_len],
                                                     generator=torch.Generator().manual_seed(42))

    val_ds.dataset.transform = tf_eval
    test_ds = datasets.ImageFolder(test_dir, transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = full_train.classes 
    return train_loader, val_loader, test_loader, class_names
