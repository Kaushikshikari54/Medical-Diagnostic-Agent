import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        # 1. Load a pre-trained ResNet18
        # ResNet18 is ideal for B.Tech projects as it is fast and accurate
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        
        # 2. Extract features
        # We take all layers EXCEPT the last fully connected (fc) layer
        # This gives us a 512-dimensional output
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        # x input shape: [batch, 3, 224, 224]
        
        # Pass through ResNet layers
        x = self.features(x) 
        # Current shape after features: [batch, 512, 1, 1]
        
        # --- THE CRITICAL FIX ---
        # We must flatten the 1x1 dimensions to make it a [batch, 512] vector
        # This allows it to be multiplied by the Bridge's linear layer
        x = torch.flatten(x, 1) 
        
        return x

    def preprocess(self, image_path):
        """
        Prepares a raw image file for the neural network.
        """
        # Standard Medical Imaging pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization based on ImageNet stats (Standard for ResNet)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            return transform(image).unsqueeze(0) # Add batch dimension: [1, 3, 224, 224]
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return None