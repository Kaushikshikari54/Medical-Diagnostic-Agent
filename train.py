import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.vision.model import VisionModel
from src.vision.bridge import ProjectionBridge
from src.vision.dataset import MedicalDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training started on {device} ---")

    # 1. Load Dataset first to find the correct number of classes
    dataset = MedicalDataset(csv_file="data/raw/metadata.csv", img_dir="data/raw")
    num_classes = len(dataset) 
    print(f"Found {num_classes} unique image-report pairs in metadata.csv")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Initialize Models with the correct dynamic dimensions
    vision_model = VisionModel().to(device)
    vision_model.eval() # We don't train the ResNet, only the Bridge
    
    bridge = ProjectionBridge(input_dim=512, output_dim=num_classes).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bridge.parameters(), lr=0.001)

    # 4. Training Loop
    for epoch in range(10): # Increased to 10 for better accuracy
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Feature extraction
            with torch.no_grad():
                features = vision_model(images)

            # Prediction
            outputs = bridge(features)
            loss = criterion(outputs, labels)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 | Avg Loss: {total_loss/len(dataloader):.4f}")

    # 5. Save the weights
    import os
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(bridge.state_dict(), "models/checkpoints/bridge_trained.pth")
    print("✅ Success: Training Complete. Weights saved.")

if __name__ == "__main__":
    train()