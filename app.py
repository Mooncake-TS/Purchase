import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly Sales Forecast (LSTM) - All SKUs", layout="wide")

# ----------------------------
# Data
# ----------------------------
@st.cache_data
def load_sales():
    df = pd.read_excel("sales.xlsx")  # 루트 기준
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

def build_model(window: int, lstm_units: int = 32):
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
        raise ValueError(f"학습 샘플이 너무 적어: {len(X)}개 (window={window}). window를 줄이거나 기간을 늘려줘.")

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

def forecast_to_target_month(model, scaler, scaled_history: np.ndarray, window: int,
                             last_month: pd.Timestamp, target_ym: str) -> int:
    """
    마지막 관측월 다음달부터 target_ym까지 재귀 예측하고,
    target_ym의 예측 판매량(int)을 반환
    """
    target = pd.to_datetime(target_ym + "-01")
    if target <= last_month:
        raise ValueError(f"예측월({target_ym})은 마지막 관측월({last_month.strftime('%Y-%m')}) 이후여야 해.")

    work = scaled_history.copy()
    months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), end=target, freq="MS")

    pred_int = 0
    for _m in months:
        lw = work[-window:].reshape(1, window, 1)
        p_sc = float(model.predict(lw, verbose=0)[0, 0])
        p_qty = float(scaler.inverse_transform([[p_sc]])[0, 0])
        pred_int = max(0, int(round(p_qty)))
        work = np.vstack([work, [[p_sc]]])
    return pred_int


# ----------------------------
# UI
# ----------------------------
st.title("📦 월 입력 → 전체 SKU 판매 예측 (LSTM)")

try:
    df = load_sales()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

skus = sorted(df["sku"].unique().tolist())
global_last_month = df["month"].max()
default_target = (global_last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

with st.sidebar:
    st.header("설정")
    target_ym = st.text_input("예측 대상 월 (YYYY-MM)", value=default_target)
    window = st.slider("입력 윈도우(개월)", 3, 24, 12)
    epochs = st.slider("학습 epochs", 50, 500, 300, step=50)
    batch_size = st.selectbox("batch size", [4, 8, 16, 32], index=1)
    seed = st.number_input("random seed", min_value=0, max_value=9999, value=42, step=1)
    show_top = st.slider("상위 N개만 보기(예측 판매량 기준)", 5, len(skus), min(20, len(skus)))

st.caption(f"현재 데이터 마지막 월: **{global_last_month.strftime('%Y-%m')}**  → 기본 예측월: **{default_target}**")

run = st.button("🚀 전체 SKU 예측 실행")

if run:
    results = []
    progress = st.progress(0)
    status = st.empty()

    # 전체 학습/예측
    for i, sku in enumerate(skus, start=1):
        status.write(f"학습/예측 중: {sku} ({i}/{len(skus)})")
        series = make_monthly_series(df, sku)
        last_month = series.index.max()

        try:
            model, scaler, scaled, _history = train_lstm(
                series.values.reshape(-1, 1),
                window=window,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )
            pred_qty = forecast_to_target_month(
                model=model,
                scaler=scaler,
                scaled_history=scaled,
                window=window,
                last_month=last_month,
                target_ym=target_ym,
            )
            results.append({"month": target_ym, "sku": sku, "forecast_sales_qty": pred_qty})
        except Exception as e:
            results.append({"month": target_ym, "sku": sku, "forecast_sales_qty": None, "error": str(e)})

        progress.progress(i / len(skus))

    status.empty()
    progress.empty()

    out = pd.DataFrame(results)

    st.subheader("✅ 전체 SKU 예측 결과")
    # 에러 있는 행은 아래로
    out_ok = out[out["forecast_sales_qty"].notna()].copy()
    out_err = out[out["forecast_sales_qty"].isna()].copy()

    out_ok = out_ok.sort_values("forecast_sales_qty", ascending=False)

    st.write(f"예측 성공: {len(out_ok)}개 / 실패: {len(out_err)}개")
    st.dataframe(out_ok.head(show_top), use_container_width=True)

    with st.expander("전체 결과(다운로드/전체표 보기)"):
        st.dataframe(out_ok, use_container_width=True)
        csv = out_ok.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV 다운로드", data=csv, file_name=f"forecast_{target_ym}.csv", mime="text/csv")

        if len(out_err) > 0:
            st.warning("실패한 SKU 목록(원인 포함)")
            st.dataframe(out_err, use_container_width=True)

    st.subheader("📊 예측 판매량 분포(히스토그램)")
    fig = plt.figure()
    plt.hist(out_ok["forecast_sales_qty"].astype(float), bins=12)
    plt.xlabel("forecast_sales_qty")
    plt.ylabel("count")
    st.pyplot(fig)

    st.info(f"📌 선택 월: {target_ym} / 전체 SKU 예측 완료")
