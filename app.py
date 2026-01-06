import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly Sales Forecast (LSTM)", layout="wide")

# ----------------------------
# Utils
# ----------------------------
@st.cache_data
def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
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
    )
    # 누락 월 처리: 0으로 채움 (원하면 interpolate로 변경 가능)
    s = s.fillna(0)
    if s.empty:
        raise ValueError(f"{sku} 데이터가 비어있어.")
    return s

def make_sequences(arr: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(X), np.array(y)

def build_model(window: int, lstm_units: int = 32) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(lstm_units),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def train_lstm(series_values: np.ndarray, window: int, epochs: int, batch_size: int, seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    X, y = make_sequences(scaled, window)
    if len(X) < 10:
        raise ValueError(
            f"LSTM 학습 샘플이 너무 적어: {len(X)}개. "
            f"window를 줄이거나(예: 6), 데이터 기간을 늘려줘."
        )

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

def forecast_recursive(model, scaler, scaled_history: np.ndarray, window: int, last_month: pd.Timestamp, target_ym: str) -> pd.DataFrame:
    target = pd.to_datetime(target_ym + "-01")

    if target <= last_month:
        raise ValueError(
            f"target_ym({target_ym})은 마지막 관측월({last_month.strftime('%Y-%m')}) 이후여야 해."
        )

    work = scaled_history.copy()
    months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), end=target, freq="MS")

    preds = []
    for m in months:
        lw = work[-window:].reshape(1, window, 1)
        p_sc = float(model.predict(lw, verbose=0)[0, 0])
        p_qty = float(scaler.inverse_transform([[p_sc]])[0, 0])
        p_int = max(0, int(round(p_qty)))

        preds.append({"month": m.strftime("%Y-%m"), "forecast_sales_qty": p_int})
        work = np.vstack([work, [[p_sc]]])  # 다음 달 예측을 위해 예측값을 히스토리에 추가

    return pd.DataFrame(preds)

# ----------------------------
# UI
# ----------------------------
st.title("📈 Monthly Sales Forecast (LSTM)")

with st.sidebar:
    st.header("설정")

    data_path = st.text_input("sales.xlsx 경로", value="sales.xlsx")
    window = st.slider("입력 윈도우(개월)", min_value=3, max_value=24, value=12, step=1)
    epochs = st.slider("학습 epochs", min_value=50, max_value=500, value=300, step=50)
    batch_size = st.selectbox("batch_size", options=[4, 8, 16, 32], index=1)
    seed = st.number_input("random seed", min_value=0, max_value=9999, value=42, step=1)

# Load data
try:
    df = load_sales(data_path)
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

skus = sorted(df["sku"].unique().tolist())

col1, col2 = st.columns([2, 1])
with col1:
    sku = st.selectbox("SKU 선택", options=skus, index=0)
with col2:
    # 기본 target을 마지막 관측월 다음달로 잡기 위해 series를 한번 만들자
    s_tmp = make_monthly_series(df, sku)
    default_target = (s_tmp.index.max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    target_ym = st.text_input("예측 대상 월(YYYY-MM)", value=default_target)

run = st.button("🚀 LSTM 학습 & 예측 실행")

if run:
    try:
        series = make_monthly_series(df, sku)
        last_month = series.index.max()
        values = series.values.reshape(-1, 1)

        with st.spinner("LSTM 학습 중... (처음 1회만 느릴 수 있어)"):
            model, scaler, scaled, history = train_lstm(
                series_values=values,
                window=window,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )

        preds = forecast_recursive(
            model=model,
            scaler=scaler,
            scaled_history=scaled,
            window=window,
            last_month=last_month,
            target_ym=target_ym,
        )

        st.success("완료!")

        st.subheader("✅ 예측 결과")
        out = preds.copy()
        out.insert(0, "sku", sku)
        st.dataframe(out, use_container_width=True)

        st.subheader("📉 학습 곡선 (loss)")
        fig1 = plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("mse")
        st.pyplot(fig1)

        st.subheader("📊 실제 판매 추이")
        fig2 = plt.figure()
        plt.plot(series.index, series.values)
        plt.xlabel("month")
        plt.ylabel("sales_qty")
        st.pyplot(fig2)

        # 마지막 달 예측값 한 줄 강조
        last_pred = int(out.iloc[-1]["forecast_sales_qty"])
        st.info(f"📌 {sku} / {target_ym} 예상 판매량: **{last_pred}**")

    except Exception as e:
        st.error(f"실행 실패: {e}")

