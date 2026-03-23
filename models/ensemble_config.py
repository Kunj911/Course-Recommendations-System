"""
Ensemble Configuration
======================
Central configuration for the multi-model ensemble.
All hyperparameters and ensemble weights live here so they are
never hardcoded across training / inference files.

Requires: scikit-learn >= 1.3.0
"""

# ── Ensemble voting weights ──────────────────────────────────────────────────

ENSEMBLE_WEIGHTS = {
    "classification": {"gb": 0.50, "rf": 0.50},
    "regression":     {"gb": 0.60, "svr": 0.40},
}

# ── Model hyperparameters ────────────────────────────────────────────────────

MODEL_PARAMS = {
    "gb_classifier": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    },
    "gb_regressor": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    },
    "rf_classifier": {
        "n_estimators": 150,
        "max_depth": 8,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "svr_regressor": {
        "kernel": "rbf",
        "C": 10,
        "epsilon": 0.3,
        "gamma": "scale",
    },
}

# ── Model file-name templates (used by trainer + loader) ─────────────────────

MODEL_FILES = {
    "gb_classifier": "{dept}_gb_classifier.pkl",
    "gb_regressor":  "{dept}_gb_regressor.pkl",
    "rf_classifier": "{dept}_rf_classifier.pkl",
    "svr_regressor": "{dept}_svr_regressor.pkl",
    "scaler":        "scaler_{dept}.pkl",
}
