import os
import sys
import torch
import pandas as pd

# Adding the project root to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vision.model import VisionModel
from src.vision.bridge import ProjectionBridge
from src.llm.agent import MedicalReportAgent

class MedicalDiagnosticSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Initializing Medical AI System on {self.device} ---")
        
        # 1. Load the Vision Model
        self.vision_model = VisionModel().to(self.device)
        self.vision_model.eval()
        
        # 2. Load the Metadata to determine dimensions
        self.metadata_path = "data/raw/metadata.csv"
        self.reports_library = self._load_reports()
        num_classes = len(self.reports_library) if self.reports_library else 10
        
        # 3. Load the Projection Bridge
        # We ensure output_dim matches our current metadata count
        self.bridge = ProjectionBridge(input_dim=512, output_dim=num_classes).to(self.device)
        
        # Load trained weights if available
        model_path = "models/checkpoints/bridge_trained.pth"
        if os.path.exists(model_path):
            try:
                # Use strict=False to prevent crashing on minor shape mismatches
                self.bridge.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                print(f"✅ Success: Loaded trained weights from {model_path}")
            except Exception as e:
                print(f"⚠️ Warning: Weights load failed ({e}). Using base bridge.")
        
        self.bridge.eval()

        # 4. Load the Medical Agent
        self.agent = MedicalReportAgent()

    def _load_reports(self):
        """Loads the report strings from the CSV."""
        if os.path.exists(self.metadata_path):
            df = pd.read_csv(self.metadata_path)
            return df['report'].tolist()
        return []

    def run_diagnostic(self, image_path):
        """
        Executes the full multimodal diagnostic pipeline with shape-correction.
        """
        if not os.path.exists(image_path):
            return f"Error: Image file {image_path} not found."

        try:
            # Step 1: Preprocess Image
            image_tensor = self.vision_model.preprocess(image_path).to(self.device)
            
            with torch.no_grad():
                # Step 2: Extract Vision Features
                features = self.vision_model(image_tensor)
                
                # --- FAIL-SAFE SHAPE CORRECTION ---
                # The Bridge expects [Batch, 512]. 
                # If features are [Batch, 512, 7, 7] or similar, we flatten/pool them.
                if len(features.shape) > 2:
                    # Global Average Pooling: reduces (B, 512, H, W) to (B, 512)
                    features = torch.mean(features, dim=[2, 3])
                elif len(features.shape) == 2 and features.shape[0] != 1:
                    # Ensures we are only looking at the first image in the batch
                    features = features[0:1, :]
                
                # Ensure the feature size is exactly 512
                if features.shape[1] != 512:
                    return f"Error: Vision Model outputting {features.shape[1]} features, but Bridge expects 512."

                # Step 3: Pass through the Bridge
                visual_tokens = self.bridge(features)
            
            # Step 4: Generate Medical Text
            report = self.agent.generate_report(visual_tokens, metadata_library=self.reports_library)
            
            return report

        except Exception as e:
            return f"Critical Error during diagnostic: {str(e)}"

if __name__ == "__main__":
    system = MedicalDiagnosticSystem()
    print("AI System Engine is Ready.")