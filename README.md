# Medical AI Diagnostic Agent 🏥

An end-to-end Multimodal Medical Imaging System that analyzes Chest X-Rays and generates automated diagnostic reports using Deep Learning.

## 🚀 Key Features
- **AI-Powered Analysis:** Uses a ResNet18 Convolutional Neural Network (CNN) for feature extraction.
- **Cross-Modal Alignment:** Features a custom Projection Bridge to map visual data to medical findings.
- **Interactive Dashboard:** Modern React frontend for seamless image uploads and real-time results.
- **High Performance:** FastAPI backend designed for low-latency model inference.

---

## 🏗️ System Architecture
The system follows a modular AI pipeline:
1. **Vision Encoder:** Extracts a 512-dimensional feature vector from X-ray images.
2. **Projection Bridge:** A linear mapping layer that aligns visual features with the medical metadata space.
3. **Report Agent:** A logic-based retrieval system that selects the most accurate medical report based on AI predictions.

---

## 📁 Project Structure
```text
Medical-diagnostic-agent/
├── data/
│   └── raw/                # Dataset: X-Ray images and metadata.csv
├── models/
│   └── checkpoints/        # Saved model weights (.pth)
├── src/
│   ├── api/                # FastAPI application (api-main.py)
│   ├── llm/                # Medical Agent logic (agent.py)
│   ├── vision/             # CNN Model, Bridge, and Dataset classes
│   └── main.py             # System Orchestrator
├── frontend/               # React UI
├── train.py                # Bridge training & Alignment script
└── requirements.txt        # Python dependencies