import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =========================
# Page
# =========================
st.set_page_config(page_title="Monthly Forecast (LSTM) - All SKUs", layout="wide")
st.title("📦 월 입력 → 전체 SKU 판매 예측 (LSTM)")
st.caption("sales.xlsx(루트) 기준. month/sku/sales_qty 컬럼 필요")

# =========================
# Data Load (ROOT)
# =========================
@st.cache_data
def load_sales_root() -> pd.DataFrame:
    df = pd.read_excel("sales.xlsx")  # ✅ app.py와 같은 폴더(루트)
    required = {"month", "sku", "sales_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales.xlsx에 컬럼이 부족해: {missing}")

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["sku", "month"]).reset_index(drop=True)
    return df

def make_monthly_series(df: pd.DataFrame, sku: str) -> pd.Series:
    # 월 시작(MS) 기준으로 월별 합계, 빠진 월은 0으로 채움
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

# =========================
# LSTM helpers
# =========================
def make_sequences(arr: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
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
    """
    series_values: shape (T, 1)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    X, y = make_sequences(scaled, window)
    if len(X) < 10:
        raise ValueError(
            f"학습 샘플이 너무 적어: {len(X)}개. (window={window}) "
            f"window를 줄이거나(예: 6) 데이터 기간을 늘려줘."
        )

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(window)

    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=0
    )

    return model, scaler, scaled, history

def forecast_to_target_month(model, scaler, scaled_history: np.ndarray, window: int,
                             last_month: pd.Timestamp, target_ym: str) -> int:
    """
    마지막 관측월 다음달부터 target_ym까지 재귀 예측 → target_ym 예측값(int) 반환
    """
    target = pd.to_datetime(target_ym + "-01")

    if target <= last_month:
        raise ValueError(
            f"예측월({target_ym})은 마지막 관측월({last_month.strftime('%Y-%m')}) 이후여야 해."
        )

    work = scaled_history.copy()
    months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        end=target,
        freq="MS"
    )

    pred_int = 0
    for _m in months:
        lw = work[-window:].reshape(1, window, 1)
        p_sc = float(model.predict(lw, verbose=0)[0, 0])
        p_qty = float(scaler.inverse_transform([[p_sc]])[0, 0])
        pred_int = max(0, int(round(p_qty)))
        work = np.vstack([work, [[p_sc]]])  # 다음 달 예측 위해 예측값을 누적

    return pred_int

# =========================
# Load data
# =========================
try:
    df = load_sales_root()
except Exception as e:
    st.error(f"❌ sales.xlsx 로드 실패: {e}")
    st.stop()

skus = sorted(df["sku"].unique().tolist())
global_last_month = df["month"].max()
default_target = (global_last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("설정")
    target_ym = st.text_input("예측 대상 월 (YYYY-MM)", value=default_target)
    window = st.slider("입력 윈도우(개월)", 3, 24, 12)
    epochs = st.slider("학습 epochs", 50, 500, 200, step=50)  # 기본 200으로 조금 빠르게
    batch_size = st.selectbox("batch size", [4, 8, 16, 32], index=1)
    seed = st.number_input("random seed", min_value=0, max_value=9999, value=42, step=1)
    top_n = st.slider("그래프/표 상위 N개만 보기", 5, len(skus), min(20, len(skus)))

st.write(f"📌 현재 데이터 마지막 월: **{global_last_month.strftime('%Y-%m')}**")
st.write(f"📌 기본 예측월: **{default_target}**")

run = st.button("🚀 전체 SKU 예측 실행")

# =========================
# Run forecast
# =========================
if run:
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, sku in enumerate(skus, start=1):
        status.write(f"학습/예측 중: {sku} ({i}/{len(skus)})")

        try:
            series = make_monthly_series(df, sku)
            last_month = series.index.max()

            model, scaler, scaled, _history = train_lstm(
                series.values.reshape(-1, 1),
                window=window,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed
            )

            pred_qty = forecast_to_target_month(
                model=model,
                scaler=scaler,
                scaled_history=scaled,
                window=window,
                last_month=last_month,
                target_ym=target_ym
            )

            results.append({
                "month": target_ym,
                "sku": sku,
                "forecast_sales_qty": pred_qty
            })

        except Exception as e:
            results.append({
                "month": target_ym,
                "sku": sku,
                "forecast_sales_qty": None,
                "error": str(e)
            })

        progress.progress(i / len(skus))

    status.empty()
    progress.empty()

    out = pd.DataFrame(results)

    # 성공/실패 분리
    out_ok = out[out["forecast_sales_qty"].notna()].copy()
    out_err = out[out["forecast_sales_qty"].isna()].copy()

    out_ok["forecast_sales_qty"] = out_ok["forecast_sales_qty"].astype(int)
    out_ok = out_ok.sort_values("forecast_sales_qty", ascending=False)

    st.subheader("✅ 예측 결과 (전체 SKU)")
    st.write(f"예측 성공: **{len(out_ok)}개** / 실패: **{len(out_err)}개**")

    st.dataframe(out_ok, use_container_width=True)

    # -------------------------
    # ✅ 원하는 그래프: x=SKU, y=수량
    # -------------------------
    st.subheader("📊 SKU별 예측 판매량 (막대그래프)")

    plot_df = out_ok.head(top_n).copy()
    plot_df = plot_df.sort_values("forecast_sales_qty", ascending=False)

    fig = plt.figure(figsize=(14, 5))
    plt.bar(plot_df["sku"], plot_df["forecast_sales_qty"])
    plt.xlabel("SKU")
    plt.ylabel("Forecast Sales Qty")
    plt.title(f"Top {top_n} SKU Forecast - {target_ym}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # 다운로드
    st.subheader("⬇️ 결과 다운로드")
    csv = out_ok.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"forecast_{target_ym}.csv",
        mime="text/csv"
    )

    if len(out_err) > 0:
        with st.expander("⚠️ 실패한 SKU (원인 보기)"):
            st.dataframe(out_err, use_container_width=True)

    st.success(f"완료! 선택 월: {target_ym}")
