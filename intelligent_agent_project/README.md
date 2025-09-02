# Intelligent Agent (Breast Cancer Classifier)

A clean, modular Python project that implements a data-driven intelligent agent using
scikit-learn on the Breast Cancer Wisconsin dataset. The agent learns to classify samples
and recommends an action based on the prediction.

## Quickstart

```bash
# 1) (Optional) Create a venv
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train + evaluate (saves model and figures in outputs/)
python -m ai_agent.cli train --test-size 0.2 --n-estimators 300

# 4) Predict
# Prepare a JSON with all feature values in the same order/names as training features.
python -m ai_agent.cli predict --json '{"mean radius": 14.5, "mean texture": 19.0, "...": 0}'
```

The command will print a JSON including the predicted label (0=benign, 1=malignant by sklearn mapping),
class probabilities, and a recommended action.

## Project Layout

- `ai_agent/` – source code (data loading, preprocessing, model, agent, evaluation, visualization, CLI).
- `outputs/` – saved model, figures (ROC, confusion matrix, feature importances), evaluation report JSON.
- `tests/` – a small smoke test.
- `requirements.txt` – dependencies.
- `README.md` – this guide.

## Notes
- Dataset is accessed via `sklearn.datasets.load_breast_cancer(as_frame=True)`.
- The *agent* object wraps the trained pipeline and exposes a `decide()` method mapping
  predictions to suggested actions.