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
st.title("📦 Inventory Planning (LSTM): FG 수량 분석 → RM 구매 → ABC")

# =========================
# Loaders (ROOT)
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
    inv["on_hand"] = pd.to_numeric(inv["on_hand"], errors="coerce").fillna(0.0)
    inv["on_order"] = pd.to_numeric(inv["on_order"], errors="coerce").fillna(0.0)
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

@st.cache_data
def load_master_root() -> pd.DataFrame:
    # ABC 분석용: sku, unit_price
    m = pd.read_excel("master.xlsx")
    m = m.copy()
    if "sku" in m.columns:
        m["sku"] = m["sku"].astype(str)
    if "unit_price" in m.columns:
        m["unit_price"] = pd.to_numeric(m["unit_price"], errors="coerce")
    return m

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
    SKU별 모델 학습 캐시: 같은 데이터/설정이면 재학습 안 함
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
# RM category rules (pattern-based)
# =========================
def rm_category(rm_sku: str) -> str:
    s = str(rm_sku).upper()
    if any(k in s for k in ["LABEL", "CAP", "BOTTLE", "CAN", "PACK", "FILM", "BOX"]):
        return "Packaging"
    if any(k in s for k in ["WATER", "CO2"]):
        return "Base"
    if any(k in s for k in ["SUGAR", "SWEET", "SYRUP"]):
        return "Sweetener"
    if any(k in s for k in ["EXTRACT", "CONC", "FRUIT"]):
        return "Extract/Concentrate"
    if any(k in s for k in ["VITAMIN", "MIX", "ADDITIVE", "FLAVOR", "ACID"]):
        return "Additive"
    return "Other"

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

    st.divider()
    top_n_fg = st.slider("FG 그래프 Top N", 5, len(skus), min(20, len(skus)))
    top_n_rm_table = st.slider("RM 테이블 Top N (net_req 기준)", 10, 200, 50)
    top_n_contrib = st.slider("RM→FG 기여도 Top N", 5, 50, 10)

    st.divider()
    st.caption("정의(합의 룰)")
    st.caption("- FG on_order = WIP")
    st.caption("- RM on_order = 발주/운송중(입고예정)")

# =========================
# Run
# =========================
run = st.button("🚀 실행")

tab1, tab2, tab3 = st.tabs(["1) 수량 분석 (FG)", "2) 원재료 구매 (RM)", "3) ABC 분석 (예측월)"])

def compute_everything():
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

    out_ok["forecast_sales_qty"] = out_ok["forecast_sales_qty"].astype(int)

    # ---------- FG compare: fg_available = on_hand + on_order(WIP) ----------
    cmp_fg = out_ok.merge(df_inv, on="sku", how="left")
    cmp_fg["on_hand"] = cmp_fg["on_hand"].fillna(0.0)
    cmp_fg["on_order"] = cmp_fg["on_order"].fillna(0.0)

    cmp_fg["fg_available_qty"] = cmp_fg["on_hand"] + cmp_fg["on_order"]  # ✅ WIP 포함
    cmp_fg["fg_need_qty"] = (cmp_fg["forecast_sales_qty"] - cmp_fg["fg_available_qty"]).clip(lower=0).round(0).astype(int)

    fg_view = cmp_fg[[
        "month", "sku", "forecast_sales_qty",
        "on_hand", "on_order", "fg_available_qty",
        "fg_need_qty"
    ]].copy().sort_values("fg_need_qty", ascending=False)

    fg_need_only = fg_view[fg_view["fg_need_qty"] > 0].copy()

    # ---------- BOM explode for RM ----------
    # fg_need_only 기준으로만 전개(생산할 게 없는 SKU는 제외)
    exp = fg_need_only[["sku", "fg_need_qty"]].rename(columns={"sku": "fg_sku"}).merge(df_bom, on="fg_sku", how="left")
    missing_bom = exp[exp["rm_sku"].isna()]["fg_sku"].unique().tolist()
    exp = exp.dropna(subset=["rm_sku"]).copy()

    exp["rm_gross_req"] = exp["fg_need_qty"] * exp["qty_per"]

    rm_gross = exp.groupby("rm_sku", as_index=False)["rm_gross_req"].sum().sort_values("rm_gross_req", ascending=False)

    # RM inventory join
    rm = rm_gross.merge(df_inv, left_on="rm_sku", right_on="sku", how="left")
    rm["on_hand"] = rm["on_hand"].fillna(0.0)
    rm["on_order"] = rm["on_order"].fillna(0.0)
    rm["rm_available"] = rm["on_hand"] + rm["on_order"]
    rm["rm_net_req"] = (rm["rm_gross_req"] - rm["rm_available"]).clip(lower=0.0)

    rm_out = rm[["rm_sku", "rm_gross_req", "on_hand", "on_order", "rm_available", "rm_net_req"]].copy()
    rm_out["rm_category"] = rm_out["rm_sku"].apply(rm_category)

    # coverage
    rm_out["coverage_ratio"] = np.where(
        rm_out["rm_gross_req"] > 0,
        rm_out["rm_available"] / rm_out["rm_gross_req"],
        np.nan
    )

    rm_out = rm_out.sort_values("rm_net_req", ascending=False).reset_index(drop=True)

    return out_ok, out_err, fg_view, fg_need_only, exp, missing_bom, rm_out

# =========================
# Execute & store in session_state
# =========================
if run:
    key = (target_ym, window, epochs, batch_size, seed)
    with st.spinner("계산 중..."):
        out_ok, out_err, fg_view, fg_need_only, exp, missing_bom, rm_out = compute_everything()
    st.session_state["result_key"] = key
    st.session_state["out_ok"] = out_ok
    st.session_state["out_err"] = out_err
    st.session_state["fg_view"] = fg_view
    st.session_state["fg_need_only"] = fg_need_only
    st.session_state["exp"] = exp
    st.session_state["missing_bom"] = missing_bom
    st.session_state["rm_out"] = rm_out
    st.success("완료!")

# =========================
# Render (if results exist)
# =========================
has_results = "out_ok" in st.session_state

with tab1:
    st.subheader("1) 수량 분석 (FG)")
    st.caption("FG on_order는 WIP로 가정하여 FG 가용재고 = on_hand + on_order")

    if not has_results:
        st.info("왼쪽 설정 후, 상단의 '실행'을 눌러줘.")
    else:
        out_ok = st.session_state["out_ok"]
        out_err = st.session_state["out_err"]
        fg_view = st.session_state["fg_view"]
        fg_need_only = st.session_state["fg_need_only"]

        st.subheader("✅ 예측 결과 (전체 SKU)")
        st.dataframe(out_ok.sort_values("forecast_sales_qty", ascending=False), use_container_width=True)

        st.subheader("🏭 예측 vs (재고+WIP) → 생산 필요량")
        st.dataframe(fg_view, use_container_width=True)

        st.subheader("📊 FG: 예측 vs 가용재고(재고+WIP) (Top N)")
        plot_fg = fg_view.sort_values("forecast_sales_qty", ascending=False).head(top_n_fg).copy()

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

        st.subheader("🧾 생산 필요 SKU만 (fg_need_qty > 0)")
        if len(fg_need_only) == 0:
            st.success("🎉 생산 필요 SKU가 없어! (예측 대비 재고+WIP가 충분)")
        else:
            st.dataframe(fg_need_only, use_container_width=True)

        if len(out_err) > 0:
            with st.expander("⚠️ 예측 실패 SKU (원인)"):
                st.dataframe(out_err, use_container_width=True)

with tab2:
    st.subheader("2) 원재료 구매 (RM)")
    st.caption("RM on_order는 발주/운송중(입고예정)으로 가정하여 RM 가용재고 = on_hand + on_order")

    if not has_results:
        st.info("먼저 '실행'을 눌러서 FG_need와 RM 계산을 만들어줘.")
    else:
        rm_out = st.session_state["rm_out"]
        exp = st.session_state["exp"]
        missing_bom = st.session_state["missing_bom"]

        if missing_bom:
            st.warning(f"BOM이 누락된 FG가 있어 RM 계산에서 제외됨: {missing_bom}")

        st.subheader("✅ RM 구매 테이블 (net_req 기준 정렬)")
        rm_table = rm_out.copy()
        rm_table_display = rm_table[[
            "rm_sku", "rm_category",
            "rm_gross_req", "rm_available",
            "on_hand", "on_order",
            "rm_net_req", "coverage_ratio"
        ]].copy()

        st.dataframe(rm_table_display.head(top_n_rm_table), use_container_width=True)

        with st.expander("전체 RM 테이블 보기"):
            st.dataframe(rm_table_display, use_container_width=True)

        st.subheader("🧩 RM 카테고리별 보기")
        cats = ["All"] + sorted(rm_table["rm_category"].unique().tolist())
        sel_cat = st.selectbox("카테고리 선택", options=cats, index=0)

        if sel_cat != "All":
            rm_cat = rm_table[rm_table["rm_category"] == sel_cat].copy()
        else:
            rm_cat = rm_table.copy()

        rm_cat_disp = rm_cat[[
            "rm_sku", "rm_category",
            "rm_gross_req", "rm_available",
            "rm_net_req"
        ]].sort_values("rm_net_req", ascending=False)

        st.dataframe(rm_cat_disp, use_container_width=True)

        st.divider()
        st.subheader("🔎 RM 하나 선택 → 어떤 FG 때문에 필요한지 (기여도)")

        rm_candidates = rm_table.sort_values("rm_net_req", ascending=False)["rm_sku"].tolist()
        if len(rm_candidates) == 0:
            st.info("RM 항목이 없어.")
        else:
            selected_rm = st.selectbox("원재료 선택", options=rm_candidates, index=0)

            # exp에는 fg_sku, fg_need_qty, rm_sku, qty_per, rm_gross_req가 있음
            contrib = exp[exp["rm_sku"] == selected_rm].copy()
            if len(contrib) == 0:
                st.info("이 원재료는 현재 생산계획 기준으로 소요가 없어.")
            else:
                contrib["fg_contrib_qty"] = contrib["fg_need_qty"] * contrib["qty_per"]
                fg_contrib = (
                    contrib.groupby("fg_sku", as_index=False)["fg_contrib_qty"]
                    .sum()
                    .sort_values("fg_contrib_qty", ascending=False)
                )
                total = fg_contrib["fg_contrib_qty"].sum()
                fg_contrib["share"] = np.where(total > 0, fg_contrib["fg_contrib_qty"] / total, np.nan)

                st.write(f"선택 RM: **{selected_rm}** | 총 소요량(gross): **{total:.2f}**")

                st.dataframe(fg_contrib.head(top_n_contrib), use_container_width=True)

                # (선택) 그래프도 같이: 너무 복잡하면 빼도 됨
                plot_c = fg_contrib.head(top_n_contrib).copy()
                figc = plt.figure(figsize=(14, 5))
                plt.bar(plot_c["fg_sku"], plot_c["fg_contrib_qty"])
                plt.xlabel("FG SKU")
                plt.ylabel("RM Contribution Qty")
                plt.title(f"FG Contribution to {selected_rm} (Top {min(top_n_contrib, len(plot_c))})")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(figc)

        # 다운로드
        st.subheader("⬇️ RM 결과 다운로드")
        csv_rm = rm_table_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "RM 구매 테이블 CSV 다운로드",
            data=csv_rm,
            file_name=f"rm_plan_{target_ym}.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("3) ABC 분석 (예측월)")
    st.caption("ABC 기준: 예측월 매출기여도 = forecast_sales_qty × unit_price")

    if not has_results:
        st.info("먼저 '실행'을 눌러 예측 수량을 만든 뒤 ABC를 계산해줘.")
    else:
        out_ok = st.session_state["out_ok"]

        # master 로드 시도
        try:
            master = load_master_root()
        except Exception as e:
            st.error(f"master.xlsx 로드 실패: {e}")
            st.stop()

        if ("sku" not in master.columns) or ("unit_price" not in master.columns):
            st.warning("master.xlsx에 sku / unit_price 컬럼이 없어서 ABC 분석을 할 수 없어.")
            st.info("master.xlsx에 최소한 (sku, unit_price) 컬럼을 추가해줘.")
        else:
            m = master[["sku", "unit_price"]].copy()
            m["sku"] = m["sku"].astype(str)
            m["unit_price"] = pd.to_numeric(m["unit_price"], errors="coerce")

            abc = out_ok.merge(m, on="sku", how="left")

            missing_price = abc[abc["unit_price"].isna()]["sku"].unique().tolist()
            if missing_price:
                st.warning(f"unit_price가 없는 SKU가 있어 ABC에서 제외됨(또는 NaN): {missing_price}")

            abc = abc.dropna(subset=["unit_price"]).copy()
            abc["forecast_value"] = abc["forecast_sales_qty"] * abc["unit_price"]

            if len(abc) == 0:
                st.error("unit_price가 매칭된 SKU가 없어 ABC 계산이 불가해.")
            else:
                abc = abc.sort_values("forecast_value", ascending=False).reset_index(drop=True)
                total_value = abc["forecast_value"].sum()
                abc["value_share"] = np.where(total_value > 0, abc["forecast_value"] / total_value, np.nan)
                abc["cum_share"] = abc["value_share"].cumsum()

                # ABC cutoffs: A 80%, B 95%, C rest
                def assign_abc(cum):
                    if cum <= 0.80:
                        return "A"
                    if cum <= 0.95:
                        return "B"
                    return "C"

                abc["abc_class"] = abc["cum_share"].apply(assign_abc)

                st.subheader("✅ ABC 결과 테이블")
                show_abc = abc[[
                    "month", "sku", "forecast_sales_qty", "unit_price",
                    "forecast_value", "cum_share", "abc_class"
                ]].copy()
                st.dataframe(show_abc, use_container_width=True)

                st.subheader("📈 파레토(누적 매출 비중)")

                # 한 figure에 bar + 누적선 (twin axis)
                x = np.arange(len(abc))
                figp = plt.figure(figsize=(14, 5))
                ax1 = plt.gca()
                ax1.bar(x, abc["forecast_value"])
                ax1.set_xlabel("SKU (sorted by forecast value)")
                ax1.set_ylabel("Forecast Value")
                ax1.set_title(f"ABC Pareto - {target_ym}")
                ax1.set_xticks([])  # SKU 라벨은 너무 많으면 안 보이니 숨김

                ax2 = ax1.twinx()
                ax2.plot(x, abc["cum_share"] * 100)
                ax2.set_ylabel("Cumulative %")

                plt.tight_layout()
                st.pyplot(figp)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("A 개수", int((abc["abc_class"] == "A").sum()))
                with col2:
                    st.metric("B 개수", int((abc["abc_class"] == "B").sum()))
                with col3:
                    st.metric("C 개수", int((abc["abc_class"] == "C").sum()))

                st.subheader("⬇️ ABC 다운로드")
                csv_abc = show_abc.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "ABC 결과 CSV 다운로드",
                    data=csv_abc,
                    file_name=f"abc_{target_ym}.csv",
                    mime="text/csv"
                )
