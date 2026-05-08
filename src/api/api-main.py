import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the system logic from your root main.py
# This assumes api-main.py is in src/api/ and main.py is in the root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from main import MedicalDiagnosticSystem

app = FastAPI(title="Medical AI Diagnostic API")

# --- CORS CONFIGURATION ---
# Necessary for the React Frontend (Port 3000) to talk to this API (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Diagnostic System
system = MedicalDiagnosticSystem()

# Ensure a temporary directory exists for uploaded files
UPLOAD_DIR = "data/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Medical AI Diagnostic API is live."}

@app.post("/diagnose")
async def diagnose_image(file: UploadFile = File(...)):
    """
    Receives an image, saves it, runs the AI pipeline, and returns the report.
    """
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 2. Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Run the complete AI pipeline
        report = system.run_diagnostic(file_path)

        return {
            "filename": file.filename,
            "diagnostic_report": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic Error: {str(e)}")

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)