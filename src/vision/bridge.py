import torch
import torch.nn as nn

class ProjectionBridge(nn.Module):
    def __init__(self, input_dim=512, output_dim=38): 
        super(ProjectionBridge, self).__init__()
        # input_dim: Features from ResNet (512)
        # output_dim: Number of unique reports in your metadata.csv
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim) # Dynamically sized to your data
        )
        
    def forward(self, x):
        return self.network(x)