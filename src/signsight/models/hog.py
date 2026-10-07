"""HOG + colour features with a class-balanced logistic regression, and the weather-condition model.

The condition input comes from the IMAGE, not from a live weather service. A
``ConditionModel`` predicts the condition (clean, fog, rain, glare) from global cues.
With ``use_condition=True`` the sign classifier gets the predicted condition
probabilities as extra features. The training probabilities come from
cross-validation, so the classifier never sees probabilities from a model that saw the same row.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..corruptions import TRAIN_CORRUPTIONS, augment
from ..features import condition_cues, sign_features

CONDITIONS = ("clean",) + TRAIN_CORRUPTIONS


class ConditionModel:
    def __init__(self, seed: int = 42) -> None:
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))

    def fit(self, images: np.ndarray, conditions: np.ndarray) -> "ConditionModel":
        self.pipe.fit(condition_cues(images), conditions)
        return self

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        proba = self.pipe.predict_proba(condition_cues(images))
        out = np.zeros((len(images), len(CONDITIONS)))
        out[:, self.pipe.classes_] = proba
        return out


class HogClassifier:
    name = "hog"

    def __init__(self, augment_share: float = 0.0, use_condition: bool = False, C: float = 0.5, seed: int = 42) -> None:
        if use_condition and augment_share <= 0:
            raise ValueError("use_condition needs augmented training data (augment_share > 0)")
        self.augment_share = augment_share
        self.use_condition = use_condition
        self.C = C
        self.seed = seed

    def _clf(self):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=self.C, class_weight="balanced", random_state=self.seed))

    def fit(self, images: np.ndarray, labels: np.ndarray, **_) -> "HogClassifier":
        cond = np.zeros(len(images), int)
        if self.augment_share > 0:
            images, labels, cond = augment(images, labels, self.augment_share, seed=self.seed)
        X = sign_features(images)
        if self.use_condition:
            self.condition_ = ConditionModel(self.seed).fit(images, cond)
            oof = cross_val_predict(ConditionModel(self.seed).pipe, condition_cues(images), cond, cv=3, method="predict_proba")
            X = np.hstack([X, oof])
        self.clf_ = self._clf().fit(X, labels)
        self.classes_ = self.clf_.classes_
        return self

    def _features(self, images: np.ndarray) -> np.ndarray:
        X = sign_features(images)
        if self.use_condition:
            X = np.hstack([X, self.condition_.predict_proba(images)])
        return X

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return self.clf_.predict_proba(self._features(images))

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(images).argmax(axis=1)]
