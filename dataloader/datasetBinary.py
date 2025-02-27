import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split

import numpy as np

SPLI={
    "train":0.8,
    "val":0.1,
    'test':0.1
}
random_state = 42
torch.manual_seed(random_state)
random.seed(random_state)
np.random.seed(random_state)


class BoxesData(Dataset):
    def __init__(self, data_path,split="train",image_size=(224,224),augemented=False):
        self.data_path=data_path
        self.split=split
        self.image_size=image_size
        self.augmented=augemented
        self.datalist=[]

        # Paths to the true and false folders
        true_folder = os.path.join(data_path, "true")
        false_folder = os.path.join(data_path, "false")


        self._add_images_from_folder(true_folder, 1)
        
        # Add images from the 'false' folder with label 0
        self._add_images_from_folder(false_folder, 0)
        
        # Split the dataset into train, val, test
        self._split_data()
        # Define image transformation
        
        if self.augmented:
            # Define image transformations (including data augmentation for training)
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size[0], self.image_size[1])),  # Resize to 224x224
                transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
                transforms.RandomRotation(20),  # Random rotation within 20 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random affine transformations (rotation, translation)
                transforms.ConvertImageDtype(torch.float),  # Convert to float tensor (0-1 range)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size[0], self.image_size[1])),  # Resize to 224x224
                transforms.ConvertImageDtype(torch.float),  # Convert to float tensor (0-1 range)
            ])
    
    def _add_images_from_folder(self, folder_path, label):
        """Add images from the specified folder to the dataset."""
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Add the image path and its corresponding label
                self.datalist.append([os.path.join(folder_path, file), label])

    def _split_data(self):
        """Split the data into train, validation, and test sets."""
        # Shuffle the dataset first
        # random.shuffle(self.datalist)
        
        # Calculate the split sizes
        train_size = 0.8
        
        # Split the data using sklearn's train_test_split
        train_data, remaining_data = train_test_split(self.datalist, train_size=train_size, random_state=random_state)
        val_data, test_data = train_test_split(remaining_data, train_size=0.5, random_state=random_state)

        # Assign the correct data split based on the 'split' parameter
        if self.split == "train":
            self.datalist = train_data
        elif self.split == "val":
            self.datalist = val_data
        elif self.split == "test":
            self.datalist = test_data

        
    def __len__(self):
        return len(self.datalist)
    

    def __getitem__(self, idx):
        img_path, label = self.datalist[idx]

        # Read the image
        img = read_image(img_path)
        if img.shape[0] == 4:
            img = img[:3, :, :]  # Remove the alpha channel (keep only RGB channels)
        # Apply the transformations
        img = self.transform(img)

        # Return the image and label
        return img, torch.tensor(label)
    