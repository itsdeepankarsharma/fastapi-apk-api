import os
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model_server import APKModelServer
from .feature_extractor import extract_features_from_apk

MODEL_PATH    = os.getenv("MODEL_PATH",    "models/gradient_boosting_tuned.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/selected_features.pkl")
SCALER_PATH   = os.getenv("SCALER_PATH",   "models/scaler.pkl")  # optional

app = FastAPI(title="APK Malware Scanner API", version="1.0.0")

# CORS: permissive for dev; restrict to your domain in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

server = APKModelServer(MODEL_PATH, FEATURES_PATH, SCALER_PATH if os.path.exists(SCALER_PATH) else None)

class FeaturePayload(BaseModel):
    features: Dict[str, float]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict-features")
def predict_from_features(payload: FeaturePayload):
    try:
        out = server.predict_from_features(payload.features)
        label_text = "malicious" if out["label"] == 0 else "benign"
        return {"label": out["label"], "label_text": label_text, "proba": out["proba"]}
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {e}")

@app.post("/predict-apk")
async def predict_from_apk(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".apk"):
        raise HTTPException(400, "Please upload an .apk file")
    apk_bytes = await file.read()
    if len(apk_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "APK too large (50 MB limit)")
    try:
        feats = extract_features_from_apk(apk_bytes, server.selected_features)
        out = server.predict_from_features(feats)
        label_text = "malicious" if out["label"] == 0 else "benign"
        return {"label": out["label"], "label_text": label_text, "proba": out["proba"], "features_populated": sum(v != 0 for v in feats.values())}
    except Exception as e:
        raise HTTPException(500, f"Failed to process APK: {e}")
