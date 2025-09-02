import json
from ai_agent.data import load_dataset
from ai_agent.preprocess import make_preprocess_pipeline, train_val_split
from ai_agent.model import make_model
from ai_agent.evaluate import evaluate_model

def test_pipeline_smoke():
    ds = load_dataset()
    X_train, X_test, y_train, y_test = train_val_split(ds.X, ds.y, test_size=0.2)
    pre = make_preprocess_pipeline(ds.feature_names)
    model = make_model(pre)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert 0.7 <= metrics["accuracy"] <= 1.0