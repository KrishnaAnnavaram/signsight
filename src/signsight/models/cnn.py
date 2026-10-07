"""A small CNN in PyTorch (extra ``torch``). Import this module only when you need it.

* weather augmentation on the fly, on training batches only
* class-weighted cross-entropy for rare classes
* early stopping on a VALIDATION part, never on the test part
* fixed seeds for torch and numpy
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..corruptions import CORRUPTIONS, TRAIN_CORRUPTIONS


def build(num_classes: int) -> nn.Module:
    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    return nn.Sequential(
        *block(3, 32), *block(32, 32), nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
        *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
    )


def _tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)


class CnnClassifier:
    name = "cnn"

    def __init__(self, epochs: int = 15, batch: int = 128, lr: float = 2e-3, augment_share: float = 0.5,
                 patience: int = 3, seed: int = 42) -> None:
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.augment_share, self.patience, self.seed = augment_share, patience, seed

    def fit(self, images, labels, val_images=None, val_labels=None) -> "CnnClassifier":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(labels)
        index = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([index[c] for c in labels])
        counts = np.bincount(y, minlength=len(self.classes_))
        weight = torch.tensor(len(y) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32)
        self.model_ = build(len(self.classes_))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        best, best_loss, bad = None, float("inf"), 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            order = rng.permutation(len(y))
            for start in range(0, len(order), self.batch):
                idx = order[start : start + self.batch]
                xb = images[idx].copy()
                for i in np.flatnonzero(rng.random(len(idx)) < self.augment_share):
                    fn = CORRUPTIONS[TRAIN_CORRUPTIONS[rng.integers(len(TRAIN_CORRUPTIONS))]]
                    xb[i] = fn(xb[i], int(rng.integers(1, 6)), rng)
                opt.zero_grad()
                loss = loss_fn(self.model_(_tensor(xb)), torch.from_numpy(y[idx]))
                loss.backward()
                opt.step()
            if val_images is not None:
                val_loss = self._loss(val_images, val_labels, loss_fn, index)
                self.history_.append({"epoch": epoch + 1, "val_loss": val_loss})
                if val_loss < best_loss - 1e-4:
                    best, best_loss, bad = copy.deepcopy(self.model_.state_dict()), val_loss, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best is not None:
            self.model_.load_state_dict(best)
        return self

    @torch.no_grad()
    def _loss(self, images, labels, loss_fn, index) -> float:
        self.model_.eval()
        y = torch.tensor([index.get(c, 0) for c in labels])
        return float(loss_fn(self._logits(images), y))

    @torch.no_grad()
    def _logits(self, images) -> torch.Tensor:
        self.model_.eval()
        return torch.cat([self.model_(_tensor(images[i : i + 512])) for i in range(0, len(images), 512)])

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return torch.softmax(self._logits(images), dim=1).numpy()

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(images).argmax(axis=1)]
