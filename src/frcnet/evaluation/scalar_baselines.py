from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(slots=True)
class FairScalarBenchmark:
    scalar_name: str
    raw_auc: float
    oriented_auc: float
    one_feature_logistic_auc: float

    @property
    def auroc(self) -> float:
        return self.raw_auc


def summarize_fair_scalar(
    *,
    scalar_name: str,
    labels: np.ndarray,
    scalar_features: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    random_state: int,
) -> FairScalarBenchmark:
    raw_auc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))
    oriented_auc = max(raw_auc, 1.0 - raw_auc)
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(scalar_features[train_index].reshape(-1, 1), labels[train_index])
    logistic_probability = classifier.predict_proba(scalar_features[test_index].reshape(-1, 1))[:, 1]
    one_feature_logistic_auc = float(roc_auc_score(labels[test_index], logistic_probability))
    return FairScalarBenchmark(
        scalar_name=scalar_name,
        raw_auc=raw_auc,
        oriented_auc=oriented_auc,
        one_feature_logistic_auc=one_feature_logistic_auc,
    )
