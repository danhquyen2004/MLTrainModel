# Mesureregression.py

import numpy as np

def NSE(y_true, y_pred):
    # Hàm NSE (Nash-Sutcliffe Efficiency)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)
