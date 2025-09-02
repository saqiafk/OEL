from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import joblib
from .utils import MODEL_DIR
from sklearn.pipeline import Pipeline

ACTIONS = {
    0: "Monitor and schedule routine checkups",
    1: "Flag for immediate diagnostic follow-up",
}

@dataclass
class AgentState:
    model_path: str

class IntelligentAgent:
    def __init__(self, model: Pipeline | None = None, state: AgentState | None = None):
        self.model = model
        self.state = state or AgentState(model_path=str(MODEL_DIR / "rf_breast_cancer.joblib"))

    def decide(self, features: Dict[str, float]) -> Dict[str, Any]:
        assert self.model is not None, "Model is not loaded."
        X = np.array([list(features.values())]).astype(float)
        proba = self.model.predict_proba(X)[0]
        label = int(self.model.predict(X)[0])
        action = ACTIONS[label]
        return {
            "predicted_label": label,
            "probabilities": proba.tolist(),
            "recommended_action": action,
        }

    def save(self):
        assert self.model is not None, "No model to save."
        joblib.dump(self.model, self.state.model_path)

    def load(self):
        self.model = joblib.load(self.state.model_path)
        return self