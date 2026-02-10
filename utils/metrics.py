import numpy as np


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def metric(pred, true):
    """Compute MAE and MSE metrics."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse
