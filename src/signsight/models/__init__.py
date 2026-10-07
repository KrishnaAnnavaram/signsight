"""Classifiers with one interface: ``fit(images, labels, ...)``, ``predict_proba(images)``, ``predict(images)``.

``HogClassifier`` is the default (scikit-learn). ``CnnClassifier`` in ``cnn.py`` needs the
extra ``torch`` and is imported only when it is used.
"""
from .hog import ConditionModel, HogClassifier

__all__ = ["ConditionModel", "HogClassifier"]
