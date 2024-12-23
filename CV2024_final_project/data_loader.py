#%%
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
#%%
class KernelDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            data_dir (str): Directory containing the blurred images and kernels
            transform (callable, optional): Optional transform to be applied on images
        """
        self.original_dir = Path("images/synthetic_original")
        self.kernel_dir = Path("images/synthetic_kernel")
        self.deformed_dir = Path("images/synthetic_deformed")


        self.transform = transform if transform is not None else transforms.ToTensor()
        
        # Get all blurred image files
        self.image_files = sorted([f for f in self.deformed_dir.glob("*.png")])
        
        print("init",len(self.image_files))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        deformed_path = self.image_files[idx]
        kernel_path = self.kernel_dir / f"{deformed_path.stem}.npy"
        original_path = self.original_dir / f"{deformed_path.stem}.png"
        
        # Load and preprocess blurred image
        blurred_img = Image.open(deformed_path).convert('L')  # Convert to grayscale
        original_img = Image.open(original_path).convert('L')
        if self.transform:
            blurred_img = self.transform(blurred_img)
            original_img = self.transform(original_img)
        
        # Load kernel
        kernel = np.load(kernel_path)
        kernel_tensor = torch.from_numpy(kernel).float()
        
        return original_img, blurred_img, kernel_tensor
            
    

def create_data_loaders(batch_size=32, train_split=0.8, num_workers=4):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (str): Directory containing the data
        batch_size (int): Batch size for the data loaders
        train_split (float): Proportion of data to use for training
        image_size (tuple): Target size for the images
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    full_dataset = KernelDataset(transform=transform)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


