import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Updated class mapping
CLASS_MAPPING = {"crosswalk": 0, "green": 1, "no": 2, "red": 3}


class TrafficSignalDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, val_split=0.2):
        """
        Args:
            data_dir (str): Path to dataset root
            split (str): One of 'train', 'valid', or 'test'
            transform (callable, optional): Optional transform to be applied on images
            val_split (float): Fraction of data to use for validation when no valid dir exists
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self.get_transforms(split == "train")

        # Check if we have a separate validation directory
        valid_dir = self.data_dir / "valid"
        if not valid_dir.exists() and split in ["train", "valid"]:
            # If no validation directory exists, use train directory and split it
            self.images_dir = self.data_dir / "train" / "images"
            self.labels_dir = self.data_dir / "train" / "labels"

            if not self.images_dir.exists():
                raise ValueError(f"Images directory not found: {self.images_dir}")
            if not self.labels_dir.exists():
                raise ValueError(f"Labels directory not found: {self.labels_dir}")

            # Get all image files
            all_files = sorted([f for f in self.images_dir.glob("*.[jJ][pP][gG]")])

            # Set random seed for reproducibility
            random.seed(42)

            # Calculate split indices
            total_files = len(all_files)
            val_size = int(total_files * val_split)

            # Randomly select validation files
            val_indices = set(random.sample(range(total_files), val_size))

            # Assign files based on split
            if split == "train":
                self.image_files = [
                    f for i, f in enumerate(all_files) if i not in val_indices
                ]
            else:  # valid
                self.image_files = [
                    f for i, f in enumerate(all_files) if i in val_indices
                ]
        else:
            # Use the specified split directory
            self.images_dir = self.data_dir / split / "images"
            self.labels_dir = self.data_dir / split / "labels"

            if not self.images_dir.exists():
                raise ValueError(f"Images directory not found: {self.images_dir}")
            if not self.labels_dir.exists():
                raise ValueError(f"Labels directory not found: {self.labels_dir}")

            self.image_files = sorted(
                [f for f in self.images_dir.glob("*.[jJ][pP][gG]")]
            )

        print(f"Found {len(self.image_files)} images in {split} set")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load and convert image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a black image and empty labels if image loading fails
            image = Image.new("RGB", (640, 640))
            return self.transform(image) if self.transform else image, torch.zeros(
                (0, 5)
            )

        # Load labels
        try:
            if label_path.exists():
                labels = []
                with open(label_path) as f:
                    for line in f.readlines():
                        try:
                            class_id, *bbox = map(float, line.strip().split())
                            labels.append([class_id] + bbox)
                        except ValueError as e:
                            print(f"Error parsing label in {label_path}: {str(e)}")
                            continue
                labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
            else:
                print(f"Warning: No label file found for {img_path.name}")
                labels = torch.zeros((0, 5))
        except Exception as e:
            print(f"Error loading labels from {label_path}: {str(e)}")
            labels = torch.zeros((0, 5))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, labels

    @staticmethod
    def get_transforms(train=True):
        """Get image transforms with augmentation for training"""
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((640, 640)),  # Maintain aspect ratio
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((640, 640)),  # Maintain aspect ratio
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )


def get_data_loaders(data_dir, batch_size=16, val_split=0.2):
    """Create train and validation data loaders"""
    # Create datasets
    train_dataset = TrafficSignalDataset(data_dir, split="train", val_split=val_split)
    val_dataset = TrafficSignalDataset(data_dir, split="valid", val_split=val_split)

    print(f"\nDataset Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader
