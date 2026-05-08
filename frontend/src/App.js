import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Activity, FileText, CheckCircle, AlertCircle } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [report, setReport] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setReport('');
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Ensure this URL matches your FastAPI server address
      const response = await axios.post('http://127.0.0.1:8000/diagnose', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setReport(response.data.diagnostic_report);
    } catch (err) {
      setError("Failed to connect to AI Backend. Ensure main.py is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f0f4f8', fontFamily: 'sans-serif', padding: '40px' }}>
      <div style={{ maxWidth: '900px', margin: '0 auto', backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', overflow: 'hidden' }}>
        
        {/* Header */}
        <div style={{ backgroundColor: '#0056b3', color: 'white', padding: '20px 30px', display: 'flex', alignItems: 'center', gap: '15px' }}>
          <Activity size={32} />
          <h1 style={{ margin: 0, fontSize: '24px' }}>Medical AI Diagnostic Agent</h1>
        </div>

        <div style={{ padding: '30px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
          
          {/* Left Column: Upload */}
          <div>
            <h3 style={{ marginTop: 0, color: '#333' }}>Step 1: Upload Chest X-Ray</h3>
            <div style={{ border: '2px dashed #cbd5e0', borderRadius: '8px', padding: '20px', textAlign: 'center', cursor: 'pointer', position: 'relative' }}>
              <input type="file" onChange={handleFileChange} style={{ opacity: 0, position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: 'pointer' }} />
              {preview ? (
                <img src={preview} alt="X-ray Preview" style={{ maxWidth: '100%', borderRadius: '4px' }} />
              ) : (
                <div style={{ color: '#718096' }}>
                  <Upload size={48} style={{ marginBottom: '10px' }} />
                  <p>Click or drag to upload JPG image</p>
                </div>
              )}
            </div>
            <button 
              onClick={analyzeImage} 
              disabled={!file || loading}
              style={{ width: '100%', marginTop: '20px', padding: '12px', backgroundColor: file ? '#0056b3' : '#a0aec0', color: 'white', border: 'none', borderRadius: '6px', fontSize: '16px', fontWeight: 'bold', cursor: file ? 'pointer' : 'not-allowed' }}
            >
              {loading ? 'Processing with AI...' : 'Run Diagnosis'}
            </button>
          </div>

          {/* Right Column: Results */}
          <div style={{ backgroundColor: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <h3 style={{ marginTop: 0, color: '#333', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileText size={20} /> AI Diagnostic Report
            </h3>
            
            <hr style={{ border: 'none', borderTop: '1px solid #e2e8f0', margin: '15px 0' }} />

            {error && (
              <div style={{ color: '#c53030', display: 'flex', gap: '8px', alignItems: 'center' }}>
                <AlertCircle size={20} /> <p>{error}</p>
              </div>
            )}

            {report ? (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#2f855a', marginBottom: '10px' }}>
                  <CheckCircle size={20} /> <strong>Diagnosis Complete</strong>
                </div>
                <p style={{ lineHeight: '1.6', color: '#2d3748', backgroundColor: 'white', padding: '15px', borderRadius: '4px', borderLeft: '4px solid #0056b3' }}>
                  {report}
                </p>
              </div>
            ) : (
              <p style={{ color: '#a0aec0', fontStyle: 'italic' }}>Upload an image and click "Run Diagnosis" to see the AI analysis.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;