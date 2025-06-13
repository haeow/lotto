import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

def load_data(xlsx_path, sequence_length=5):
    df = pd.read_excel(xlsx_path)
    data = df.iloc[:, 1:7].values  # 번호1~6만 사용

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def to_onehot(y):
    onehot = np.zeros((len(y), 45))
    for i, row in enumerate(y):
        for n in row:
            onehot[i, n - 1] = 1
    return onehot