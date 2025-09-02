import argparse, json
from .data import load_dataset
from .preprocess import make_preprocess_pipeline, train_val_split
from .model import make_model, ModelConfig
from .evaluate import evaluate_model
from .visualize import plot_roc, plot_confusion_matrix, plot_feature_importance
from .agent import IntelligentAgent
from .utils import REPORT_DIR

def cmd_train(args):
    ds = load_dataset()
    X_train, X_test, y_train, y_test = train_val_split(ds.X, ds.y, test_size=args.test_size)
    pre = make_preprocess_pipeline(ds.feature_names)
    model = make_model(pre, ModelConfig(n_estimators=args.n_estimators, max_depth=args.max_depth))
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    agent = IntelligentAgent(model=model)
    agent.save()

    roc_path = plot_roc(model, X_test, y_test)
    cm_path = plot_confusion_matrix(model, X_test, y_test)
    fi_path = plot_feature_importance(model, ds.feature_names)

    report = {
        "metrics": metrics,
        "figures": {"roc_curve": str(roc_path), "confusion_matrix": str(cm_path), "feature_importance": str(fi_path)},
    }
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

def cmd_predict(args):
    ds = load_dataset()
    agent = IntelligentAgent().load()
    if args.json_file:
        data = json.loads(open(args.json_file, "r").read())
    else:
        data = json.loads(args.json)
    features = {k: float(data[k]) for k in ds.feature_names}
    decision = agent.decide(features)
    print(json.dumps(decision, indent=2))

def main():
    p = argparse.ArgumentParser(description="Intelligent Agent CLI")
    sub = p.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train and evaluate the model")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--max-depth", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Run inference with the saved agent model")
    p_predict.add_argument("--json", type=str, help="Inline JSON with feature_name: value pairs")
    p_predict.add_argument("--json-file", type=str, help="Path to JSON file with feature_name: value pairs")
    p_predict.set_defaults(func=cmd_predict)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()