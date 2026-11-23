"""Train and evaluate the BillWise models.

Run:  python train.py

Two models are trained:
  Model A - appliance answers only          (for new users with no bill history)
  Model B - appliance answers + last month's kWh  (for returning users)

Every algorithm learns a household CORRECTION FACTOR (actual kWh / physics
estimate), see models_ratio.py. Prediction = physics estimate x factor, so
appliance changes always move the bill in the right direction.

For each, several algorithms are compared with repeated 5-fold cross-validation
against a physics baseline. The best one (lowest MAE) is saved.
"""
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import config
from features import BASE_FEATURES, HISTORY_FEATURES, load_dataset
from models_ratio import CorrectionFactorRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42


def base_algorithms():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6,
                                               min_samples_leaf=2, random_state=SEED),
    }


def candidate_models(use_history=False):
    """Each algorithm learns the correction factor (actual / physics kWh)."""
    return {name: CorrectionFactorRegressor(m, use_history=use_history)
            for name, m in base_algorithms().items()}


def scores(y_true, y_pred):
    return {
        "MAE_kWh": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE_kWh": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 1),
    }


def evaluate(model, X, y):
    """Repeated 5-fold CV: average scores over 3 different shuffles."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    maes, rmses, r2s, mapes = [], [], [], []
    for train_idx, test_idx in rkf.split(X):
        m = model.fit(X.iloc[train_idx], y.iloc[train_idx])
        s = scores(y.iloc[test_idx].values, m.predict(X.iloc[test_idx]))
        maes.append(s["MAE_kWh"]); rmses.append(s["RMSE_kWh"])
        r2s.append(s["R2"]); mapes.append(s["MAPE_%"])
    return {"MAE_kWh": round(np.mean(maes), 1), "MAE_std": round(np.std(maes), 1),
            "RMSE_kWh": round(np.mean(rmses), 1), "R2": round(np.mean(r2s), 3),
            "MAPE_%": round(np.mean(mapes), 1)}


def train_one(name, df, features, target, use_history=False):
    X, y = df[features], df[target]
    print(f"\n=== {name}: {len(X)} households, {len(features)} features ===")

    results = {"Physics baseline (no ML)": scores(y.values, df["physics_kwh"].values)}
    print(f"{'Physics baseline (no ML)':<26} MAE {results['Physics baseline (no ML)']['MAE_kWh']:>6} kWh")
    for algo, model in candidate_models(use_history).items():
        results[algo] = evaluate(model, X, y)
        r = results[algo]
        print(f"{algo:<26} MAE {r['MAE_kWh']:>6} ± {r['MAE_std']:<5} kWh  "
              f"RMSE {r['RMSE_kWh']:>6}  R² {r['R2']:>6}  MAPE {r['MAPE_%']}%")

    trained = {k: v for k, v in results.items() if not k.startswith("Physics")}
    best_name = min(trained, key=lambda k: trained[k]["MAE_kWh"])
    best = candidate_models(use_history)[best_name]
    print(f"--> Best: {best_name}")

    # Out-of-fold predictions give honest residuals for the prediction range
    oof = cross_val_predict(best, X, y, cv=KFold(5, shuffle=True, random_state=SEED))
    residuals = y.values - oof
    interval = [float(np.percentile(residuals, 10)), float(np.percentile(residuals, 90))]

    best.fit(X, y)   # final model is trained on ALL data
    return {"model": best, "algorithm": best_name, "features": features,
            "interval": interval, "results": results, "oof": oof, "y": y.values}


def importance(bundle):
    m = bundle["model"]   # importance of context features for the correction factor
    return dict(sorted(zip(m.context_features, map(float, m.feature_importances_)),
                       key=lambda t: -t[1]))


def plot(bundle, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(bundle["y"], bundle["oof"], alpha=0.7)
    lo, hi = bundle["y"].min(), bundle["y"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual kWh"); ax.set_ylabel("Predicted kWh (cross-validated)")
    ax.set_title(f"{title}: {bundle['algorithm']}")
    imp = importance(bundle)
    names = list(imp)[:10][::-1]
    axes[1].barh(names, [imp[n] for n in names])
    axes[1].set_title("Correction-factor features")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    df = load_dataset(config.DATA_PATH, config.DATA_SHEET)
    df["prev_kwh"] = df["kwh_sep"]
    df = df.dropna(subset=["prev_kwh"])
    print(f"Loaded {len(df)} households from {config.DATA_PATH}")

    model_a = train_one("Model A (no history)", df, BASE_FEATURES, "kwh_oct")
    model_b = train_one("Model B (with last month)", df, HISTORY_FEATURES, "kwh_oct", use_history=True)

    plot(model_a, "Model A", os.path.join(config.REPORT_DIR, "model_a.png"))
    plot(model_b, "Model B", os.path.join(config.REPORT_DIR, "model_b.png"))

    bundle = {}
    for key, b in (("a", model_a), ("b", model_b)):
        bundle[key] = {k: b[k] for k in ("model", "algorithm", "features", "interval")}
    joblib.dump(bundle, config.MODEL_PATH)

    metrics = {
        key: {"algorithm": b["algorithm"], "interval_kwh": b["interval"],
              "results": b["results"], "feature_importance": importance(b)}
        for key, b in (("model_a", model_a), ("model_b", model_b))
    }
    metrics["n_households"] = len(df)
    with open(os.path.join(config.REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({k: v["results"] for k, v in (("Model A", model_a), ("Model B", model_b))}) \
        .to_json(os.path.join(config.REPORT_DIR, "comparison.json"), indent=2)

    print(f"\nSaved model  -> {config.MODEL_PATH}")
    print(f"Saved report -> {config.REPORT_DIR}/metrics.json, model_a.png, model_b.png")


if __name__ == "__main__":
    main()
