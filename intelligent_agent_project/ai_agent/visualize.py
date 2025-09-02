from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from .utils import FIG_DIR

def plot_roc(model, X_test, y_test, fname: Optional[str] = None):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    out = FIG_DIR / (fname or "roc_curve.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_confusion_matrix(model, X_test, y_test, fname: Optional[str] = None):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    out = FIG_DIR / (fname or "confusion_matrix.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_feature_importance(model, feature_names, fname: Optional[str] = None):
    importances = model.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=90)
    plt.title("Feature Importances")
    plt.tight_layout()
    out = FIG_DIR / (fname or "feature_importance.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out