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

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dense(45, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def train_model(xlsx_path):
    X, y = load_data(xlsx_path)
    y_onehot = to_onehot(y)

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y_onehot, epochs=20, batch_size=16)
    model.save("lotto_model.h5")
    print("✅ 모델 학습 완료 및 저장")
    
def predict_numbers(xlsx_path):
    if not os.path.exists("lotto_model.h5"):
        print("❌ 먼저 모델을 학습해주세요 (train_model 호출 필요)")
        return

    model = load_model("lotto_model.h5")
    X, _ = load_data(xlsx_path)

    latest_seq = X[-1].reshape(1, 5, 6)
    pred = model.predict(latest_seq)[0]

    top6_indices = pred.argsort()[-6:][::-1]
    main_numbers = sorted([i + 1 for i in top6_indices])

    remaining = [i for i in range(45) if i not in top6_indices]
    bonus_index = max(remaining, key=lambda i: pred[i])
    bonus_number = bonus_index + 1

    print("🎯 추천 번호 (6개):", main_numbers)
    print("🎁 보너스 번호:", bonus_number)