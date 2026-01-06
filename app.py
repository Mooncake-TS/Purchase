import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly Sales Forecast (LSTM)", layout="wide")

# =========================
# 🔹 데이터 로드 (루트 기준)
# =========================
@st.cache_data
def load_sales():
    df = pd.read_excel("sales.xlsx")  # ✅ 루트에서 바로 읽음
    required = {"month", "sku", "sales_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales.xlsx에 컬럼이 부족해: {missing}")

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["sku", "month"]).reset_index(drop=True)
    return df

def make_monthly_series(df: pd.DataFrame, sku: str) -> pd.Series:
    s = (
        df[df["sku"] == sku]
        .groupby("month")["sales_qty"]
        .sum()
        .sort_index()
        .asfreq("MS")
        .fillna(0)
    )
    if s.empty:
        raise ValueError(f"{sku} 데이터가 비어있어.")
    return s

def make_sequences(arr: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(X), np.array(y)

def build_model(window: int):
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def train_lstm(series_values: np.ndarray, window: int, epochs: int, batch_size: int):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    X, y = make_sequences(scaled, window)
    if len(X) < 10:
        raise ValueError("데이터가 너무 짧아 LSTM 학습이 어려워.")

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(window)

    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=0,
    )

    return model, scaler, scaled, history

def forecast_recursive(model, scaler, scaled_history, window, last_month, target_ym):
    target = pd.to_datetime(target_ym + "-01")
    work = scaled_history.copy()

    months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        end=target,
        freq="MS"
    )

    preds = []
    for m in months:
        lw = work[-window:].reshape(1, window, 1)
        p_sc = model.predict(lw, verbose=0)[0, 0]
        p_qty = scaler.inverse_transform([[p_sc]])[0, 0]
        p_int = max(0, int(round(p_qty)))

        preds.append({"month": m.strftime("%Y-%m"), "forecast_sales_qty": p_int})
        work = np.vstack([work, [[p_sc]]])

    return pd.DataFrame(preds)

# =========================
# 🔹 UI
# =========================
st.title("📈 Monthly Sales Forecast (LSTM)")

df = load_sales()
skus = sorted(df["sku"].unique())

with st.sidebar:
    st.header("설정")
    window = st.slider("입력 윈도우(개월)", 3, 24, 12)
    epochs = st.slider("학습 epochs", 50, 500, 300, step=50)
    batch_size = st.selectbox("batch size", [4, 8, 16, 32], index=1)

sku = st.selectbox("SKU 선택", skus)

series = make_monthly_series(df, sku)
last_month = series.index.max()
default_target = (last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

target_ym = st.text_input("예측 대상 월 (YYYY-MM)", value=default_target)

if st.button("🚀 LSTM 예측 실행"):
    with st.spinner("LSTM 학습 중..."):
        model, scaler, scaled, history = train_lstm(
            series.values.reshape(-1, 1),
            window,
            epochs,
            batch_size
        )

    preds = forecast_recursive(
        model, scaler, scaled, window, last_month, target_ym
    )

    st.success("완료!")

    st.subheader("예측 결과")
    preds.insert(0, "sku", sku)
    st.dataframe(preds, use_container_width=True)

    st.subheader("실제 판매 추이")
    fig = plt.figure()
    plt.plot(series.index, series.values)
    plt.xlabel("month")
    plt.ylabel("sales_qty")
    st.pyplot(fig)

    st.info(
        f"📌 {sku} / {target_ym} 예상 판매량: "
        f"**{int(preds.iloc[-1]['forecast_sales_qty'])}**"
    )
