"""Correction-factor model used by BillWise.

Idea: final kWh = physics estimate x correction factor.
  - The physics estimate (appliances.py) reacts to every appliance input,
    so changing AC temperature, hours, showers etc. always moves the bill
    in the right direction.
  - The ML model learns the correction factor (actual kWh / physics kWh)
    from household context: how much this kind of household really uses
    compared with what its answers suggest (e.g. people overestimate AC hours).

Only CONTEXT features (not the ones a recommendation changes) go into the
correction model, so a "Try it" change affects the bill only through the
physics estimate, which keeps it consistent with the saving shown on the tip.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

CONTEXT_FEATURES = ["occupants", "n_ac", "ac_hp", "heater", "heater_instant", "all_home"]
HISTORY_CONTEXT = CONTEXT_FEATURES + ["prev_ratio"]


class CorrectionFactorRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model=None, use_history=False, clip=(0.3, 2.5)):
        self.base_model = base_model
        self.use_history = use_history
        self.clip = clip

    def _context(self, X):
        X = pd.DataFrame(X).copy()
        if self.use_history:
            X["prev_ratio"] = X["prev_kwh"] / X["physics_kwh"].clip(lower=1)
        cols = HISTORY_CONTEXT if self.use_history else CONTEXT_FEATURES
        return X[cols]

    def fit(self, X, y):
        ratio = np.clip(np.asarray(y, float) / X["physics_kwh"].clip(lower=1).values, *self.clip)
        self.model_ = clone(self.base_model).fit(self._context(X), np.log(ratio))
        return self

    def predict_factor(self, X):
        return np.clip(np.exp(self.model_.predict(self._context(X))), *self.clip)

    def predict(self, X):
        return X["physics_kwh"].values * self.predict_factor(X)

    @property
    def feature_importances_(self):
        m = self.model_
        if hasattr(m, "feature_importances_"):
            return m.feature_importances_
        coef = np.abs(m[-1].coef_)
        return coef / coef.sum()

    @property
    def context_features(self):
        return HISTORY_CONTEXT if self.use_history else CONTEXT_FEATURES
