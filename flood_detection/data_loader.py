import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class FloodNetDataset(Dataset):
    """
    Custom PyTorch Dataset for the FloodNet aerial imagery dataset.
    Loads images and maps them to binary labels:
    - 0: Not Flooded
    - 1: Flooded
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initializes the dataset with image paths, labels, and optional transforms.

        Args:
            image_paths (list of str): Absolute paths to the images.
            labels (list of int): Integer class labels (0 or 1) for the images.
            transform (callable, optional): Torchvision transform to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in this dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the preprocessed image tensor and label at the specified index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to standard 3-channel RGB
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_inference_transform():
    """
    Returns the torchvision transformation pipeline required for single-image inference.
    Resizes images to 224x224 and normalizes them with ImageNet mean and std.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def get_dataloaders(data_dir, batch_size=32, split_ratio=(0.70, 0.15, 0.15), random_seed=42):
    """
    Scans the data directory, maps image names to labels, splits the dataset into
    stratified train, validation, and test subsets, and returns corresponding DataLoaders.

    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int): Batch size for loaders.
        split_ratio (tuple): Target split ratios for (train, val, test).
        random_seed (int): Seed for reproducibility of train/val/test splits.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    all_image_paths = []
    
    # Recursively find all image paths
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
    if not all_image_paths:
        raise ValueError(f"No images found in '{data_dir}'. Make sure your images are placed there.")
        
    image_paths = []
    labels = []
    
    # Infer labels from directory names or paths
    for path in all_image_paths:
        path_lower = path.lower()
        # Exclude mask labels (which are grayscale mask PNG files)
        if "_lab.png" in path_lower or "/mask" in path_lower or "/label/" in path_lower:
            continue
            
        # Check for non-flooded tags first (since "non-flooded" contains "flooded")
        if any(sub in path_lower for sub in ["non-flooded", "non_flooded", "not-flooded", "not_flooded"]):
            labels.append(0)
            image_paths.append(path)
        elif "flooded" in path_lower:
            labels.append(1)
            image_paths.append(path)
            
    if not image_paths:
        raise ValueError(
            "Could not parse classification labels from image filenames or folder structures. "
            "Ensure that image paths contain either 'Flooded' or 'Non-Flooded' as directories."
        )
        
    train_ratio, val_ratio, test_ratio = split_ratio
    temp_ratio = val_ratio + test_ratio  # 0.30
    
    # Stratified split requires at least 2 instances of each class
    from collections import Counter
    class_counts = Counter(labels)
    stratify_labels = labels if all(count >= 2 for count in class_counts.values()) else None
    
    # Split into train and a temporary validation/test dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_labels
    )
    
    # Split the temporary dataset into validation and test (50% each of the 30% temp subset)
    val_size_relative = val_ratio / temp_ratio
    temp_class_counts = Counter(y_temp)
    stratify_temp = y_temp if all(count >= 2 for count in temp_class_counts.values()) else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1.0 - val_size_relative,
        random_state=random_seed,
        stratify=stratify_temp
    )
    
    print("Dataset split completed successfully:")
    print(f"  Total classification images: {len(image_paths)}")
    print(f"  Train size: {len(X_train)} (Flooded: {sum(y_train)}, Not Flooded: {len(y_train) - sum(y_train)})")
    print(f"  Val size:   {len(X_val)} (Flooded: {sum(y_val)}, Not Flooded: {len(y_val) - sum(y_val)})")
    print(f"  Test size:  {len(X_test)} (Flooded: {sum(y_test)}, Not Flooded: {len(y_test) - sum(y_test)})")
    
    # Load transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_test_transform = get_inference_transform()
    
    # Datasets
    train_dataset = FloodNetDataset(X_train, y_train, transform=train_transform)
    val_dataset = FloodNetDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = FloodNetDataset(X_test, y_test, transform=val_test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
