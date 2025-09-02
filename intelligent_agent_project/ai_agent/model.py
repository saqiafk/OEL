from dataclasses import dataclass
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

@dataclass
class ModelConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    random_state: int = 42
    class_weight: str | None = "balanced"

def make_model(preprocessor, cfg: ModelConfig | None = None) -> Pipeline:
    cfg = cfg or ModelConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        class_weight=cfg.class_weight,
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    return pipe