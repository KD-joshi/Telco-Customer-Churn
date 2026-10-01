"""
Pipeline construction utilities.

Builds full scikit-learn Pipeline objects that handle every step from raw
DataFrame input → prediction, including type fixing, feature engineering,
scaling, encoding, and classification. This guarantees zero data leakage
between train and test because all transformations are fitted only on
training folds.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from preprocessing import FeatureEngineer, TotalChargesFixer


def _build_preprocessor():
    """Construct a ColumnTransformer that handles numeric and categorical
    columns independently.

    Numeric path:  impute missing → standardize
    Categorical path: impute missing → one-hot encode
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_logistic_pipeline():
    """Baseline: Logistic Regression with L2 regularization.

    class_weight='balanced' handles the ~73/27 class imbalance by
    adjusting the loss to penalize minority-class misclassification more
    heavily — no manual resampling needed.
    """
    return Pipeline(
        steps=[
            ("type_fixer", TotalChargesFixer()),
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight=1.0):
    """XGBoost gradient-boosted tree classifier.

    scale_pos_weight compensates for class imbalance by weighting the
    positive (churn) class proportionally to its under-representation.
    """
    return Pipeline(
        steps=[
            ("type_fixer", TotalChargesFixer()),
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
