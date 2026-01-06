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
st.set_page_config(page_title="Inventory Planning (LSTM)", layout="wide")
st.title("📦 Inventory Planning: 수량 분석 → 원재료 구매")

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
    inv["on_hand"] = pd.to_numeric(inv["on_hand"], errors="coerce").fillna(0).astype(float)
    inv["on_order"] = pd.to_numeric(inv["on_order"], errors="coerce").fillna(0).astype(float)
    return inv

@st.cache_data
def load_bom_root() -> pd.DataFrame:
    bom = pd.read_excel("BOM.xlsx")
    required = {"fg_sku", "rm_sku", "qty_per"}
    missing = required - set(bom.columns)
    if missing:
        raise ValueError(f"BOM.xlsx에 컬럼이 부족해: {missing}")

    bom = bom.copy()
    bom["fg_sku"] = bom["fg_sku"].astype(str)
    bom["rm_sku"] = bom["rm_sku"].astype(str)
    bom["qty_per"] = pd.to_numeric(bom["qty_per"], errors="coerce").fillna(0.0)
    return bom

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
    SKU별 모델 학습 캐시: 같은 설정으로 다시 실행하면 학습 시간 크게 단축.
    """
    series_values = np.array(series_values_tuple, dtype=float).reshape(-1, 1)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    X, y = make_sequences(scaled, window)
    if len(X) < 10:
        raise ValueError(f"학습 샘플이 너무 적어: {len(X)}개 (window={window})")

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
    마지막 관측월 다음달부터 target_ym까지 재귀 예측 → target_ym 예측값(int)
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
        work = np.vstack([work, [[p_sc]]])

    return pred_int

# =========================
# Load files
# =========================
try:
    df_sales = load_sales_root()
    df_inv = load_inventory_root()
    df_bom = load_bom_root()
except Exception as e:
    st.error(f"❌ 파일 로드 실패: {e}")
    st.stop()

skus = sorted(df_sales["sku"].unique().tolist())
global_last_month = df_sales["month"].max()
default_target = (global_last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("공통 설정")
    target_ym = st.text_input("예측 대상 월 (YYYY-MM)", value=default_target)
    window = st.slider("입력 윈도우(개월)", 3, 24, 12)
    epochs = st.slider("학습 epochs", 50, 300, 100, step=50)
    batch_size = st.selectbox("batch size", [4, 8, 16, 32], index=1)
    seed = st.number_input("random seed", min_value=0, max_value=9999, value=42, step=1)

    top_n_fg = st.slider("FG 그래프 Top N", 5, len(skus), min(20, len(skus)))
    top_n_rm = st.slider("RM 그래프 Top N", 5, 50, 20)

    st.divider()
    st.caption("정의(합의한 룰)")
    st.caption("- FG on_order = WIP (생산중/완성 예정)")
    st.caption("- RM on_order = 발주/운송중(입고 예정)")

# =========================
# Run button
# =========================
run = st.button("🚀 실행")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["1) 수량 분석 (FG)", "2) 원재료 구매 (RM)"])

if run:
    # ---------- Forecast all SKUs ----------
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, sku in enumerate(skus, start=1):
        status.write(f"LSTM 예측 중: {sku} ({i}/{len(skus)})")
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

            results.append({"month": target_ym, "sku": sku, "forecast_sales_qty": int(pred_qty)})

        except Exception as e:
            results.append({"month": target_ym, "sku": sku, "forecast_sales_qty": None, "error": str(e)})

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

    # ---------- FG compare: fg_available = on_hand + on_order(WIP) ----------
    inv_fg = df_inv.copy()  # FG/RM 섞여 있어도 sku로만 merge하면 됨
    cmp_fg = out_ok.merge(inv_fg, on="sku", how="left")
    cmp_fg["on_hand"] = cmp_fg["on_hand"].fillna(0)
    cmp_fg["on_order"] = cmp_fg["on_order"].fillna(0)

    cmp_fg["fg_available_qty"] = cmp_fg["on_hand"] + cmp_fg["on_order"]  # ✅ on_order를 WIP로 사용
    cmp_fg["fg_need_qty"] = (cmp_fg["forecast_sales_qty"] - cmp_fg["fg_available_qty"]).clip(lower=0)

    # float 가능(원하면 int로 바꿔도 됨)
    cmp_fg["fg_need_qty"] = cmp_fg["fg_need_qty"].round(0).astype(int)

    # ============================================================
    # TAB 1: 수량 분석 (FG)
    # ============================================================
    with tab1:
        st.subheader("✅ 예측 결과 (전체 SKU)")
        st.dataframe(out_ok.sort_values("forecast_sales_qty", ascending=False), use_container_width=True)

        st.subheader("🏭 FG: 예측 vs (재고 + WIP) 비교 → 생산 필요량")
        show_fg = cmp_fg[["sku", "forecast_sales_qty", "on_hand", "on_order", "fg_available_qty", "fg_need_qty"]].copy()
        show_fg = show_fg.sort_values("fg_need_qty", ascending=False)
        st.dataframe(show_fg, use_container_width=True)

        st.subheader("📊 FG: 예측 vs 가용재고(재고+WIP) (Top N)")
        plot_fg = show_fg.sort_values("forecast_sales_qty", ascending=False).head(top_n_fg).copy()

        x = np.arange(len(plot_fg))
        width = 0.42
        fig1 = plt.figure(figsize=(14, 5))
        plt.bar(x - width/2, plot_fg["forecast_sales_qty"], width=width, label="Forecast")
        plt.bar(x + width/2, plot_fg["fg_available_qty"], width=width, label="FG Available (On hand + WIP)")
        plt.xticks(x, plot_fg["sku"], rotation=45, ha="right")
        plt.xlabel("FG SKU")
        plt.ylabel("Qty")
        plt.title(f"Forecast vs FG Available - {target_ym} (Top {top_n_fg})")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig1)

        st.subheader("🧾 생산 필요 SKU만")
        fg_need_only = show_fg[show_fg["fg_need_qty"] > 0].copy()
        if len(fg_need_only) == 0:
            st.success("🎉 생산 필요 SKU가 없어! (예측 대비 재고+WIP가 충분)")
        else:
            st.dataframe(fg_need_only, use_container_width=True)

        if len(out_err) > 0:
            with st.expander("⚠️ 예측 실패 SKU (원인)"):
                st.dataframe(out_err, use_container_width=True)

    # ============================================================
    # TAB 2: 원재료 구매 (RM)  ← 너가 원하는 핵심: '원재료 부족량 그래프 1개'
    # ============================================================
    with tab2:
        st.subheader("🧪 원재료 부족량 그래프 (이번 달 구매해야 할 원재료)")

        fg_need = cmp_fg[["sku", "fg_need_qty"]].rename(columns={"sku": "fg_sku"}).copy()
        fg_need = fg_need[fg_need["fg_need_qty"] > 0]

        if len(fg_need) == 0:
            st.info("생산 필요량이 0이라 원재료 구매도 필요 없어.")
        else:
            # BOM explode
            exp = fg_need.merge(df_bom, on="fg_sku", how="left")
            missing_bom = exp[exp["rm_sku"].isna()]["fg_sku"].unique().tolist()
            if missing_bom:
                st.warning(f"BOM이 없는 FG가 있어 정전개에서 제외됨: {missing_bom}")

            exp = exp.dropna(subset=["rm_sku"]).copy()
            exp["rm_gross_req"] = exp["fg_need_qty"] * exp["qty_per"]

            rm_gross = exp.groupby("rm_sku", as_index=False)["rm_gross_req"].sum()
            rm_gross = rm_gross.sort_values("rm_gross_req", ascending=False)

            # RM inventory join (on_hand + on_order = available)
            rm = rm_gross.merge(df_inv, left_on="rm_sku", right_on="sku", how="left")
            rm["on_hand"] = rm["on_hand"].fillna(0)
            rm["on_order"] = rm["on_order"].fillna(0)
            rm["rm_available"] = rm["on_hand"] + rm["on_order"]

            # net requirement (shortage) = 구매 필요량
            rm["rm_net_req"] = (rm["rm_gross_req"] - rm["rm_available"]).clip(lower=0)

            # 보기 좋은 정리
            rm_out = rm[["rm_sku", "rm_gross_req", "on_hand", "on_order", "rm_available", "rm_net_req"]].copy()
            rm_out = rm_out.sort_values("rm_net_req", ascending=False)

            # 부족만 남기기
            rm_short = rm_out[rm_out["rm_net_req"] > 0].copy()

            if len(rm_short) == 0:
                st.success("🎉 원재료가 충분해! (총소요량 대비 재고+입고예정이 커버)")
                st.dataframe(rm_out, use_container_width=True)
            else:
                # ✅ 너가 원한 '원재료 부족량' 단일 그래프
                plot_rm = rm_short.head(top_n_rm).copy()

                fig2 = plt.figure(figsize=(14, 5))
                plt.bar(plot_rm["rm_sku"], plot_rm["rm_net_req"])
                plt.xlabel("RM SKU")
                plt.ylabel("Net Requirement (Purchase Qty)")
                plt.title(f"RM Net Requirement (Purchase Needed) - {target_ym} (Top {min(top_n_rm, len(plot_rm))})")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig2)

                st.subheader("🧾 원재료 부족 목록 (구매 필요)")
                st.dataframe(rm_short, use_container_width=True)

                # 다운로드
                csv_rm = rm_short.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "원재료 구매 필요량 CSV 다운로드",
                    data=csv_rm,
                    file_name=f"rm_purchase_{target_ym}.csv",
                    mime="text/csv"
                )
