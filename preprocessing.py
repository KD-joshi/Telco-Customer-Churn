"""
Data loading and preprocessing utilities.

Handles:
- Raw CSV ingestion and type coercion (TotalCharges string → float)
- Missing-value imputation for new customers (tenure=0, TotalCharges blank)
- scikit-learn–compatible transformers for use inside Pipeline objects
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import DATA_PATH, ID_COL, TARGET, NUMERIC_FEATURES


def load_dataset(path=None):
    """Load the raw CSV and return features DataFrame and target Series."""
    path = path or DATA_PATH
    df = pd.read_csv(path)

    # TotalCharges is stored as string; 11 rows have whitespace instead of a
    # number (all correspond to tenure == 0, i.e. brand-new customers).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # For tenure-0 customers the sensible imputation is 0.0 — they simply
    # haven't been billed yet.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target as integer
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COL])

    return X, y


class TotalChargesFixer(BaseEstimator, TransformerMixin):
    """Pipeline-safe transformer that coerces TotalCharges to float.

    During inference the field may arrive as a string from JSON input.
    This transformer ensures it is always numeric before downstream steps.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(
                X["TotalCharges"], errors="coerce"
            ).fillna(0.0)
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of DataFrame columns by name."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives new features from raw columns based on EDA observations.

    New features:
    - avg_monthly_spend:  TotalCharges / max(tenure, 1) — captures whether a
      customer's current plan is pricier than their historical average.
    - num_services:       count of add-on services the customer subscribes to.
      Bundled services create switching costs that reduce churn.
    - has_support:        binary flag for OnlineSecurity or TechSupport — EDA
      showed these are strong retention anchors.
    - tenure_charge_interaction: tenure × MonthlyCharges — captures the joint
      effect of loyalty and spend level.
    """

    _SERVICE_COLS = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Average monthly spend — highlights customers paying more now
        # than their historical average (potential dissatisfaction signal)
        tenure_safe = X["tenure"].replace(0, 1)
        X["avg_monthly_spend"] = X["TotalCharges"] / tenure_safe

        # Count of subscribed add-on services (Yes = subscribed)
        X["num_services"] = sum(
            (X[col] == "Yes").astype(int) for col in self._SERVICE_COLS
            if col in X.columns
        )

        # Whether customer has any protective/support service
        if "OnlineSecurity" in X.columns and "TechSupport" in X.columns:
            X["has_support"] = (
                (X["OnlineSecurity"] == "Yes") | (X["TechSupport"] == "Yes")
            ).astype(int)

        # Interaction: loyal high-spenders vs. new high-spenders behave
        # very differently w.r.t. churn
        X["tenure_charge_interaction"] = X["tenure"] * X["MonthlyCharges"]

        return X
