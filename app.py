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
st.set_page_config(page_title="Forecast vs ERP Inventory (LSTM)", layout="wide")
st.title("📦 월 입력 → 전체 SKU 예측(LSTM) → ERP 재고 비교")
st.caption("루트에 sales.xlsx / inventory.xlsx 필요 (month, sku, sales_qty) / (sku, on_hand, on_order)")

# =========================
# Data Load (ROOT)
# =========================
@st.cache_data
def load_sales_root() -> pd.DataFrame:
    df = pd.read_excel("sales.xlsx")
    required = {"month", "sku", "sales_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales.xlsx에 컬럼이 부족해: {missing}")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["sku"] = df["sku"].astype(str)
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["sku", "month"]).reset_index(drop=True)
    return df

@st.cache_data
def load_inventory_root() -> pd.DataFrame:
    inv = pd.read_excel("inventory.xlsx")
    required = {"sku", "on_hand", "on_order"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"inventory.xlsx에 컬럼이 부족해: {missing}")

    inv = inv.copy()
    inv["sku"] = inv["sku"].astype(str)
    inv["on_hand"] = pd.to_numeric(inv["on_hand"], errors="coerce").fillna(0).astype(int)
    inv["on_order"] = pd.to_numeric(inv["on_order"], errors="coerce").fillna(0).astype(int)
    return inv

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

@st.cache_resource
def train_lstm_cached(series_values_tuple, window: int, epochs: int, batch_size: int, seed: int):
    """
    SKU별 모델 학습을 캐시해서 같은 설정으로 재실행 시 시간을 크게 줄임.
    series_values_tuple: 캐시 키 안정화를 위한 튜플 입력
    """
    series_values = np.array(series_values_tuple, dtype=float).reshape(-1, 1)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    X, y = make_sequences(scaled, window)
    if len(X) < 10:
        raise ValueError(
            f"학습 샘플이 너무 적어: {len(X)}개 (window={window}). "
            f"window를 줄이거나 데이터 기간을 늘려줘."
        )

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(window)

    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=0
    )

    return model, scaler, scaled

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
        work = np.vstack([work, [[p_sc]]])  # 다음 step을 위해 누적

    return pred_int

# =========================
# Load data
# =========================
try:
    df_sales = load_sales_root()
except Exception as e:
    st.error(f"❌ sales.xlsx 로드 실패: {e}")
    st.stop()

try:
    df_inv = load_inventory_root()
except Exception as e:
    st.error(f"❌ inventory.xlsx 로드 실패: {e}")
    st.stop()

skus = sorted(df_sales["sku"].unique().tolist())
global_last_month = df_sales["month"].max()
default_target = (global_last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("설정")
    target_ym = st.text_input("예측 대상 월 (YYYY-MM)", value=default_target)
    window = st.slider("입력 윈도우(개월)", 3, 24, 12)
    epochs = st.slider("학습 epochs", 50, 300, 100, step=50)  # 기본 빠르게
    batch_size = st.selectbox("batch size", [4, 8, 16, 32], index=1)
    seed = st.number_input("random seed", min_value=0, max_value=9999, value=42, step=1)
    top_n = st.slider("그래프 Top N", 5, len(skus), min(20, len(skus)))
    show_all_table = st.checkbox("전체 비교 테이블도 펼쳐서 보기", value=False)

st.write(f"📌 현재 데이터 마지막 월: **{global_last_month.strftime('%Y-%m')}**")
st.write(f"📌 기본 예측월: **{default_target}**")

run = st.button("🚀 예측 & ERP 비교 실행")

# =========================
# Run
# =========================
if run:
    results = []
    progress = st.progress(0)
    status = st.empty()

    # --- Forecast all SKUs ---
    for i, sku in enumerate(skus, start=1):
        status.write(f"LSTM 학습/예측 중: {sku} ({i}/{len(skus)})")

        try:
            series = make_monthly_series(df_sales, sku)
            last_month = series.index.max()

            series_tuple = tuple(series.values.tolist())
            model, scaler, scaled = train_lstm_cached(
                series_values_tuple=series_tuple,
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
                "forecast_sales_qty": int(pred_qty)
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
    out_ok = out[out["forecast_sales_qty"].notna()].copy()
    out_err = out[out["forecast_sales_qty"].isna()].copy()

    if len(out_ok) == 0:
        st.error("예측 성공한 SKU가 없어. window/데이터 기간을 확인해줘.")
        st.stop()

    out_ok["forecast_sales_qty"] = out_ok["forecast_sales_qty"].astype(int)
    out_ok = out_ok.sort_values("forecast_sales_qty", ascending=False)

    st.subheader("✅ 예측 결과")
    st.dataframe(out_ok, use_container_width=True)

    # --- Compare with ERP inventory ---
    cmp = out_ok.merge(df_inv, on="sku", how="left")
    cmp["on_hand"] = cmp["on_hand"].fillna(0).astype(int)
    cmp["on_order"] = cmp["on_order"].fillna(0).astype(int)
    cmp["available_qty"] = cmp["on_hand"] + cmp["on_order"]
    cmp["shortage_qty"] = (cmp["forecast_sales_qty"] - cmp["available_qty"]).clip(lower=0).astype(int)

    cmp = cmp[[
        "month", "sku",
        "forecast_sales_qty",
        "on_hand", "on_order", "available_qty",
        "shortage_qty"
    ]].sort_values("shortage_qty", ascending=False)

    # =========================
    # Graph 1: Forecast by SKU (Top N)
    # =========================
    st.subheader("📊 SKU별 예측 판매량 (Top N)")
    plot_f = out_ok.head(top_n).copy().sort_values("forecast_sales_qty", ascending=False)

    fig1 = plt.figure(figsize=(14, 5))
    plt.bar(plot_f["sku"], plot_f["forecast_sales_qty"])
    plt.xlabel("SKU")
    plt.ylabel("Forecast Sales Qty")
    plt.title(f"Top {top_n} SKU Forecast - {target_ym}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

    # =========================
    # Graph 2: Forecast vs Available (Top N)
    # =========================
    st.subheader("📊 SKU별 예측 vs 가용재고(ERP) 비교 (Top N)")

    cmp_plot = cmp.copy()
    # 비교 그래프는 예측이 큰 SKU 위주로 보여주는 게 보통 더 직관적
    cmp_plot = cmp_plot.sort_values("forecast_sales_qty", ascending=False).head(top_n)

    x = np.arange(len(cmp_plot))
    width = 0.42

    fig2 = plt.figure(figsize=(14, 5))
    plt.bar(x - width/2, cmp_plot["forecast_sales_qty"], width=width, label="Forecast")
    plt.bar(x + width/2, cmp_plot["available_qty"], width=width, label="Available (On hand + On order)")
    plt.xticks(x, cmp_plot["sku"], rotation=45, ha="right")
    plt.xlabel("SKU")
    plt.ylabel("Qty")
    plt.title(f"Forecast vs Available - {target_ym} (Top {top_n})")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # =========================
    # Table: Shortage only
    # =========================
    st.subheader("🧾 부족 수량 테이블 (Shortage > 0)")
    shortage_table = cmp[cmp["shortage_qty"] > 0].copy()

    if len(shortage_table) == 0:
        st.success("🎉 부족 SKU가 없어! 예측 대비 재고/입고예정이 충분해.")
    else:
        st.dataframe(shortage_table, use_container_width=True)

    # 전체 테이블 옵션
    if show_all_table:
        st.subheader("📋 전체 비교 테이블 (Forecast vs ERP)")
        st.dataframe(cmp, use_container_width=True)

    # 다운로드
    st.subheader("⬇️ 결과 다운로드")
    csv_cmp = cmp.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Forecast vs ERP 비교 결과 CSV 다운로드",
        data=csv_cmp,
        file_name=f"forecast_vs_erp_{target_ym}.csv",
        mime="text/csv"
    )

    if len(out_err) > 0:
        with st.expander("⚠️ 예측 실패 SKU (원인)"):
            st.dataframe(out_err, use_container_width=True)

    st.success(f"완료! 선택 월: {target_ym}")
