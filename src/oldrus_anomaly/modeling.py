from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["ngram_surp", "ft_outlier", "log_freq", "char_len"]


def train_token_classifier(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Обучает простой классификатор token-anomaly (замены).

    Используется логистическая регрессия + стандартизация.
    """
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    return clf


def extract_feature_matrix(rows: List[Dict]) -> np.ndarray:
    """rows -> матрица признаков в порядке FEATURE_COLS"""
    X = np.zeros((len(rows), len(FEATURE_COLS)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(FEATURE_COLS):
            X[i, j] = float(r[c])
    return X


def feature_weights(clf: Pipeline) -> Dict[str, float]:
    """Возвращает веса признаков логистической регрессии (для отчёта)."""
    lr: LogisticRegression = clf.named_steps["lr"]
    coef = lr.coef_.reshape(-1)
    return {name: float(w) for name, w in zip(FEATURE_COLS, coef)}
