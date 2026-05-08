import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        # This transform is the "missing link" - it converts PIL images to Tensors
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image name and report
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx, 0])
        
        # Load image
        image = Image.open(img_name).convert('RGB')
        
        # Apply the transformation (Crucial!)
        if self.transform:
            image = self.transform(image)
        
        # Return the image tensor and the index (label)
        # We use the index as a numeric label for the CrossEntropyLoss
        return image, idx