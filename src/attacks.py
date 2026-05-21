import numpy as np
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier, SklearnClassifier


def wrap_sklearn(model, nb_classes: int) -> SklearnClassifier:
    # ART >= 1.17 SklearnClassifier infers nb_classes from the model; no kwarg needed
    return SklearnClassifier(model=model)


def wrap_pytorch(
    model: nn.Module,
    input_shape: tuple,
    nb_classes: int,
    device: str = "cpu",
) -> PyTorchClassifier:
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=device,
    )


def run_fgsm(classifier, X: np.ndarray, eps: float) -> np.ndarray:
    return FastGradientMethod(estimator=classifier, eps=eps).generate(x=X)


def run_pgd(
    classifier,
    X: np.ndarray,
    eps: float,
    max_iter: int = 20,
    eps_step=None,
) -> np.ndarray:
    if eps_step is None:
        eps_step = eps / 10
    return ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
    ).generate(x=X)


def apply_constraints(X_adv: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    X_adv = np.nan_to_num(X_adv, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(X_adv, X_train.min(axis=0), X_train.max(axis=0))
