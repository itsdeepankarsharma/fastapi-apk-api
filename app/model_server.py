import joblib
import pandas as pd
from typing import Dict, Optional

class APKModelServer:
    def __init__(self, model_path: str, features_path: str, scaler_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        self.selected_features = joblib.load(features_path)  # ordered list of feature names
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def _frame_from_features(self, features: Dict[str, float]) -> pd.DataFrame:
        row = {f: float(features.get(f, 0.0)) for f in self.selected_features}
        return pd.DataFrame([row], columns=self.selected_features)

    def predict_from_features(self, features: Dict[str, float]) -> Dict:
        X = self._frame_from_features(features)
        Xv = self.scaler.transform(X.values) if self.scaler is not None else X.values
        y = self.model.predict(Xv)[0]
        proba = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(Xv)[0]
            proba = {"class0": float(p[0]), "class1": float(p[1])}
        return {"label": int(y), "proba": proba}
    