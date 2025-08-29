# app.py
# 실행: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

# Plotly optional import (없어도 기본 차트로 폴백)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False

# --------------------------
# 공용 유틸
# --------------------------
def line_chart(df, x, y, color=None, title="", hover_data=None):
    if HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, color=color, title=title, hover_data=hover_data)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        if title:
            st.subheader(title)
        if color and color in df.columns:
            for k, sub in df.groupby(color):
                st.line_chart(sub.set_index(x)[y], height=320)
        else:
            st.line_chart(df.set_index(x)[y], height=320)

def _rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[m]-y_pred[m])**2))) if m.sum() else np.nan

def _mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[m]-y_pred[m]))) if m.sum() else np.nan

def _mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    return float(np.mean(np.abs((y_true[m]-y_pred[m])/y_true[m]))*100) if m.sum() else np.nan

# --------------------------
# 카테고리 데이터 로더
# --------------------------
def _load_category_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 날짜 컬럼 탐지: time(YYYY-MM) 또는 date
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
    elif "date" in df.columns:
        # 다양한 포맷 허용
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    else:
        raise ValueError("CSV에 'time' 또는 'date' 컬럼이 필요합니다.")
    return df.sort_values("date").reset_index(drop=True)

# --------------------------
# 페이지 세팅
# --------------------------
st.set_page_config(page_title="CCSI Storytelling Dashboard", page_icon="📈", layout="wide")
st.title("📈 CCSI 예측 스토리텔링 대시보드")

# --------------------------
# 사이드바 (분석 설정)
# --------------------------
st.sidebar.header("⚙️ 분석 설정")
date_start = st.sidebar.date_input("시작일", value=datetime(2022,1,1))
date_end   = st.sidebar.date_input("종료일", value=datetime(2025,6,1))

# 기본값 설정 (CSV 결과 사용 모드)
use_results = True
results_path = "results/ccsi_total2.csv"

# --------------------------
# 탭 구성
# --------------------------
tabs = st.tabs([
    "Intro",
    "Step 1: 기존 CCSI 예측(데모 즉시 표시)",
    "Step 2: 분해 접근(대/소분류)",
    "Step 3: 대분류 예측",
    "Step 4: 소분류 예측",
    "Step 5: 성능 비교",
    "Final Insight"
])

# --------------------------
# Intro
# --------------------------
with tabs[0]:
        # --- Intro: 추가 정보 요약 카드/설명 ---
        st.markdown("### 📅 데이터")
        st.markdown("""
        - **기간**: 2022-01 ~ 2025-06 (일별)
        - **지역**: 경기도 **일부 시** (카드 소비), 전국 **CCSI**
        - **소비액**: 대분류(예: 식음료, 의류) 및 소분류(예: 커피, 패스트푸드)및 연령대별 시간대별 소비액과 소비건수 집계 데이터                    
        """)

        with st.expander("🧹 전처리 핵심 보기", expanded=False):
            st.markdown("""
            **시계열 공통**
            - 결측치 처리: 앞/뒤 채움 + 카테고리 평균/계절 평균 보완
            - 이상치 완화: IQR/표준편차 기준 감쇠 또는 윈저라이징

            **피처 엔지니어링 (누설 방지를 위해 모두 `shift(1)` 적용)**
            - Lag: `lag_1`, `lag_2`, `lag_3`, `lag_12`
            - Windowing(rolling): `roll_mean_3/6/12`, `roll_std_3/6/12`, `roll_min/max`
            - 추세/기울기: 롤링 구간 선형회귀 **slope**, 변화율(ROC), 모멘텀
            - 계절성/더미: `month`, `quarter`, 명절/프로모션/정책 더미
            - 상호작용: (카테고리 × 계절) 교호항
            - 외생 변수: 물가·금리·실업률·카드건수 등 **lag 적용** 후 결합

            **타깃 전처리**
            - 차분: 1차/계절(12) 차분으로 비정상성 완화 (필요 시)
            - 역변환: 예측 후 역차분/역변환 로직 명시적으로 관리

            **데이터 분리/검증**
            - 홀드아웃: 최근 N개월(예: 6개월) 고정
            - 그룹별 학습: `category_l1/category_l2`별 개별 모델 또는 공유 하이퍼파라미터
            - 시계열 분할: 시간 보존(split) · 누락 없는 누적 학습
            - **데이터 누설 방지**: 미래값 참조 금지(모든 파생은 과거 정보로만)
            """)

        st.markdown("### 📏 평가 프로토콜")
        st.markdown("""
        - **홀드아웃**: 최근 N개월(예: 6개월) 고정  
        - **지표**: RMSE·MAE·MAPE (규모/해석 용이성 균형)  
        - 주석: 합산 성능은 *카테고리 합산 → CCSI 근사* 관점에서 함께 해석
        """)

        with st.expander("⚠️ 해석 시 유의점", expanded=False):
            st.markdown("""
            - 표본 편향: **경기도 일부 시** 카드 데이터 → 전국 소비심리와 차이 가능  
            - 커버리지: 현금/비카드, 특정 업종 미포함 가능성  
            - 구성 지표 차이: **CCSI는 설문 기반 심리지수**, 소비액은 실거래 → 시차/탄력성 차이 존재
            """)

        st.markdown("### 🧭 사용 가이드")
        st.markdown("""
        1. 좌측 **기간 필터**를 설정한다.  
        2. **Step 1**에서 CCSI 실제/예측과 **오차**의 규모를 파악한다.  
        3. **Step 2**로 이동해 **대분류별** 예측 성능과 패턴을 비교한다.  
        4. **Step 3**에서 관심 대분류의 **소분류**로 드릴다운해 민감도를 본다.  
        5. (선택) 향후 Step 5에서 **합산 성능 → CCSI 근사**를 검토한다.
        """)
    

# --------------------------
# Step 1: 파일 업로드 없이 바로 그래프 표시 (데모)
# --------------------------
with tabs[1]:
    if use_results:
        st.subheader("기존 CCSI 단일 예측 시도 (결과 CSV)")
        try:
            df = pd.read_csv(results_path)
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            st.stop()

        # 필요한 컬럼 확인
        req_cols = {"time", "y_true", "y_pred"}
        if not req_cols.issubset(df.columns):
            st.error("필수 컬럼(time,y_true,y_pred)이 없습니다.")
            st.stop()

        # time을 YYYY-MM으로 파싱
        df["date"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # ⏱️ 기간 필터(사이드바) 적용
        try:
            start_d = pd.to_datetime(date_start)
            end_d = pd.to_datetime(date_end)
        except Exception:
            start_d = df["date"].min()
            end_d = df["date"].max()
        mask = (df["date"] >= start_d) & (df["date"] <= end_d)
        df = df.loc[mask].copy()
        # CCSI는 전체 구간 유지, 예측값은 2024-01 이후만 표시
        cutoff = pd.to_datetime("2024-01-01")
        df.loc[df["date"] < cutoff, "y_pred"] = np.nan

        # 지표 (필터 적용 후)
        rmse = _rmse(df["y_true"], df["y_pred"])
        mae  = _mae(df["y_true"], df["y_pred"])
        mape = _mape(df["y_true"], df["y_pred"])

        c1,c2,c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.2f}")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("MAPE(%)", f"{mape:.1f}%")

       

        # 라인 차트 (실제 vs 예측)
        plot_df_final = df.rename(columns={"y_true":"Actual", "y_pred":"Pred"})[["date","Actual","Pred"]]
        plot_df_final = plot_df_final.melt("date", var_name="series", value_name="value")
        if HAS_PLOTLY:
            import plotly.express as px
            fig = px.line(plot_df_final, x="date", y="value", color="series", title="Actual vs. Pred (CCSI)")
            # 선+마커로 보기 좋게
            fig.update_traces(mode="lines+markers")
            # Trace 색상 지정
            fig.for_each_trace(
                lambda tr: tr.update(line=dict(color="blue")) if tr.name == "Actual" else tr.update(line=dict(color="red"))
            )
            # Hover 템플릿 한글 커스텀 with error
            merged = df[["date","y_true","y_pred"]].copy()
            merged["error"] = merged["y_pred"] - merged["y_true"]
            fig.for_each_trace(
                lambda tr: tr.update(
                    customdata=merged[["error"]].values,
                    hovertemplate=("실제:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
                ) if tr.name == "Actual" else tr.update(
                    customdata=merged[["error"]].values,
                    hovertemplate=("예측:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
                )
            )
            fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 폴백: 기본 라인차트 (커스텀 hover 불가)
            line_chart(plot_df_final, x="date", y="value", color="series", title="Actual vs. Pred (CCSI)")

        # (선택) 추가 차트: Parity & Residuals (간단 버전)
        if HAS_PLOTLY:
            import plotly.express as px
            parity = px.scatter(df, x="y_true", y="y_pred", title="Parity Plot (y_true vs y_pred)")
            st.plotly_chart(parity, use_container_width=True)
        else:
            st.caption("Plotly 미설치: Parity Plot 생략")

         # Step1 피처 엔지니어링 설명
        with st.expander("🔧 Step 1에 사용한 피처 엔지니어링(요약)", expanded=False):
            st.markdown(
                """
                **데이터 단위**: 월별(YYYY-MM)

                #### 📌 공통 규칙
                - 모든 파생 피처는 **미래 누설 방지**를 위해 *과거값 기준*(`shift`)으로 생성

                #### 🟦 CCSI 기반 피처
                - **지연값(Lag)**: `CCSI_lag_1`, `CCSI_lag_2`, `CCSI_lag_3`, `CCSI_lag_12`
                - **롤링 통계(6개월)**: `CCSI_roll_mean_6`, `CCSI_roll_std_6`
                - **차분**: `CCSI_diff_1`, `CCSI_diff_12`
                - **추세(기울기)**: `CCSI_slope_3` *(최근 3개월 선형회귀 기울기)*

                #### 🟩 카드 소비액 기반 피처(총액)
                - **지연값(Lag)**: `총액_lag_1`, `총액_lag_2`, `총액_lag_3`, `총액_lag_12`
                - **롤링 통계(6개월)**: `총액_roll_mean_6`, `총액_roll_std_6`, `총액_roll_max_6`, `총액_roll_min_6`
                - **차분**: `총액_diff_1`, `총액_diff_12`
                - **추세(기울기)**: `총액_slope_3`

                #### 🗓️ 캘린더
                - `month` (1~12), 필요 시 

                ---
                아래는 실제 컬럼 스냅샷 예시입니다.
                """
            )
            st.code(
                """연월, CCSI, 연월_dt, 총액_lag_1, CCSI_lag_1, 총액_lag_2, CCSI_lag_2, 총액_lag_3, CCSI_lag_3, 총액_lag_12, CCSI_lag_12, 총액_roll_mean_6, 총액_roll_std_6, CCSI_roll_mean_6, CCSI_roll_std_6, 총액_roll_max_6, 총액_roll_min_6, 총액_diff_1, CCSI_diff_1, 총액_diff_12, CCSI_diff_12, 총액_slope_3, CCSI_slope_3, month\n202201, 104.9, 2022-01-01, ..., 1""",
                language="text",
            )

# --------------------------
# Step 2: 분해 접근 (대분류)
# --------------------------
with tabs[2]:
    st.subheader("대분류 분해 접근 (설명)")

    st.markdown(
        """
        **왜 분해가 필요한가?**  
        단일 **CCSI + 소비액** 예측은 업종별 민감도·구성비 변화 때문에 오차가 커집니다.  
        **대분류/소분류 단위로 각각 예측 → 합산**하면, 업종별 리드-랙 구조와 계절성을 더 잘 포착할 수 있습니다.
        """
    )

    st.markdown("### 대분류 목록 (9개)")
    st.markdown(
        "- 공공/기업/단체  \n"
        "- 공연/전시  \n"
        "- 미디어/통신  \n"
        "- 생활서비스  \n"
        "- 소매/유통  \n"
        "- 여가/오락  \n"
        "- 음식  \n"
        "- 의료/건강  \n"
        "- 학문/교육"
    )

    st.markdown("### 접근 개요")
    st.markdown(
        """
        1. **대분류 단위 예측**: 업종별 모델(또는 하나의 멀티타스크 모델)로 월별 소비액을 예측합니다.  
        2. **소분류 정밀화(선택)**: 대분류 내 소분류까지 세분화하여 예측 후 대분류로 집계합니다.  
        3. **합산 및 CCSI 근사**: (대/소분류 예측) 합산치를 이용해 CCSI 추정 혹은 동행/선행 분석을 수행합니다.
        """
    )

    st.markdown("### 수식")
    st.latex(r"\text{TotalConsumption}_t = \sum_{k=1}^{K} \text{Cons}_{t}^{(k)}")
    st.caption("의미: 시점 t의 **전체 소비액**은 모든 업종 k의 소비액을 **합산**한 값입니다.")
    st.latex(r"\text{Cons}_{t}^{(k)} = f_k(\text{CCSI}_{t-\ell}, \text{Lag/Window}, \text{Seasonality}, \text{Exogenous})")
    st.caption(r"의미: 업종 k의 소비액은 **과거 CCSI**, **Lag/Window 피처**, **계절성**, **외생 변수**를 입력으로 하는 모델 \(f_k\)로 **설명/예측**됩니다.")

    st.markdown("### 평가 지표")
    c1,c2,c3 = st.columns(3)
    c1.metric("권장 지표", "RMSE")
    c2.metric("보조 지표", "MAE")
    c3.metric("상대 오차", "MAPE (%)")
    

# --------------------------
# Step 3: 대분류 예측 (9개 + 전체 성능)
# --------------------------
with tabs[3]:
    st.subheader("대분류별 CCSI 예측 (9개) 및 전체 성능")

    # CSV 로드: 고정 파일명만 사용
    try:
        df_l1 = pd.read_csv("results/ccsi_firstgrade.csv")
    except Exception:
        st.error("CSV를 찾을 수 없습니다. 프로젝트 폴더의 'ccsi_firstgrade.csv'를 확인해주세요.")
        st.stop()

    # 날짜 파싱: time(YYYY-MM) 또는 date 허용
    if "date" in df_l1.columns:
        try:
            df_l1["date"] = pd.to_datetime(df_l1["date"], errors="coerce")
        except Exception:
            df_l1["date"] = pd.to_datetime(df_l1["date"].astype(str), errors="coerce")
    elif "time" in df_l1.columns:
        df_l1["date"] = pd.to_datetime(df_l1["time"], format="%Y-%m", errors="coerce")
    else:
        st.error("CSV에는 'time'(YYYY-MM) 또는 'date' 컬럼이 필요합니다.")
        st.stop()

    # 필수 컬럼 확인: category_l1, y_true, y_pred
    req_cols_l1 = {"category_l1", "y_true", "y_pred"}
    if not req_cols_l1.issubset(df_l1.columns):
        # 👉 WIDE 형식 감지: 'CCSI'와 '*_pred' 계열 열이 있으면 LONG 변환 시도
        pred_cols = [c for c in df_l1.columns if c.endswith("_pred_MA3") or c.endswith("_pred")]
        if ("CCSI" in df_l1.columns) and len(pred_cols) > 0:
            # id_vars 구성 (존재하는 컬럼만)
            id_vars = [c for c in ["time", "date", "CCSI"] if c in df_l1.columns]
            # wide → long
            df_l1 = df_l1.melt(
                id_vars=id_vars,
                value_vars=pred_cols,
                var_name="category_l1",
                value_name="y_pred"
            )
            # 실제값 컬럼명 표준화
            if "CCSI" in df_l1.columns and "y_true" not in df_l1.columns:
                df_l1 = df_l1.rename(columns={"CCSI": "y_true"})
            # 대분류명에서 접미어 제거
            df_l1["category_l1"] = (
                df_l1["category_l1"]
                .str.replace("_pred_MA3", "", regex=False)
                .str.replace("_pred", "", regex=False)
            )
        else:
            st.error("필수 컬럼이 없습니다. (category_l1, y_true, y_pred) 또는 WIDE 형식(‘CCSI’ + '*_pred*')이 아닙니다.")
            st.stop()

    # 기간 필터(사이드바) 적용
    try:
        start_d = pd.to_datetime(date_start)
        end_d = pd.to_datetime(date_end)
    except Exception:
        start_d = df_l1["date"].min()
        end_d = df_l1["date"].max()
    m_l1 = (df_l1["date"] >= start_d) & (df_l1["date"] <= end_d)
    df_l1 = df_l1.loc[m_l1].copy()
    # CCSI는 전체 구간 유지, 예측값은 2024-01 이후만 표시
    cutoff = pd.to_datetime("2024-01-01")
    df_l1.loc[df_l1["date"] < cutoff, "y_pred"] = np.nan

    # 전체(모든 대분류 합친 관측치) 기준 성능 지표
    overall_rmse = _rmse(df_l1["y_true"], df_l1["y_pred"])
    overall_mae  = _mae(df_l1["y_true"], df_l1["y_pred"])
    overall_mape = _mape(df_l1["y_true"], df_l1["y_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 RMSE", f"{overall_rmse:.2f}")
    c2.metric("전체 MAE", f"{overall_mae:.2f}")
    c3.metric("전체 MAPE(%)", f"{overall_mape:.1f}%")

    st.divider()

    # 대분류 선택 UI
    l1_list = sorted(df_l1["category_l1"].dropna().unique().tolist())
    if not l1_list:
        st.warning("대분류(category_l1) 값이 없습니다.")
        st.stop()

    pick = st.selectbox("대분류 선택 (9개)", l1_list)

    sub_l1 = df_l1[df_l1["category_l1"] == pick].copy()

    # 선택한 대분류 성능
    l1_rmse = _rmse(sub_l1["y_true"], sub_l1["y_pred"])
    l1_mae  = _mae(sub_l1["y_true"], sub_l1["y_pred"])
    l1_mape = _mape(sub_l1["y_true"], sub_l1["y_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"[{pick}] RMSE", f"{l1_rmse:.2f}")
    c2.metric(f"[{pick}] MAE", f"{l1_mae:.2f}")
    c3.metric(f"[{pick}] MAPE(%)", f"{l1_mape:.1f}%")

    # 라인 차트 (선택한 대분류: 실제 vs 예측)
    plot_df_l1 = sub_l1.rename(columns={"y_true":"Actual", "y_pred":"Pred"})[["date","Actual","Pred"]]
    plot_df_l1_m = plot_df_l1.melt("date", var_name="series", value_name="value")

    if HAS_PLOTLY:
        fig_l1 = px.line(plot_df_l1_m, x="date", y="value", color="series", title=f"[{pick}] Actual vs Pred (대분류)")
        fig_l1.update_traces(mode="lines+markers")
        fig_l1.for_each_trace(
            lambda tr: tr.update(line=dict(color="blue")) if tr.name == "Actual" else tr.update(line=dict(color="red"))
        )
        merged_l1 = sub_l1[["date","y_true","y_pred"]].copy()
        merged_l1["error"] = merged_l1["y_pred"] - merged_l1["y_true"]
        fig_l1.for_each_trace(
            lambda tr: tr.update(
                customdata=merged_l1[["error"]].values,
                hovertemplate=("실제:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            ) if tr.name == "Actual" else tr.update(
                customdata=merged_l1[["error"]].values,
                hovertemplate=("예측:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            )
        )
        fig_l1.update_xaxes(dtick="M1", tickformat="%Y-%m")
        st.plotly_chart(fig_l1, use_container_width=True)
    else:
        line_chart(plot_df_l1_m, x="date", y="value", color="series", title=f"[{pick}] Actual vs Pred (대분류)")

    st.divider()

    # 9개 대분류 성능 테이블
    rows_l1 = []
    for g, df_g in df_l1.groupby("category_l1"):
        rows_l1.append({
            "category_l1": g,
            "RMSE": _rmse(df_g["y_true"], df_g["y_pred"]),
            "MAE":  _mae(df_g["y_true"], df_g["y_pred"]),
            "MAPE(%)": _mape(df_g["y_true"], df_g["y_pred"]),
        })
    comp_l1 = pd.DataFrame(rows_l1).sort_values("RMSE")
    st.markdown("**대분류별 성능 비교 (9개)**")
    st.dataframe(comp_l1, use_container_width=True)

# --------------------------
# Step 4: 소분류 예측 (대분류→소분류 선택 + 전체 평균 지표)
# --------------------------
with tabs[4]:
    st.subheader("소분류 Drill-down & 예측")

    # 0) 소분류 예측 파일 로드 (XLSX 고정)
    sec_path = Path("results") / "ccsi_secgrade.xlsx"
    try:
        # 모든 시트를 읽어와 합칩니다 (시트=각 소분류 예측)
        xls_all = pd.read_excel(sec_path, sheet_name=None)
        frames = []
        for sh_name, df_sh in xls_all.items():
            if df_sh is None or len(df_sh) == 0:
                continue
            df_tmp = df_sh.copy()
            # 각 시트 이름 보존 → 소분류 선택에 사용
            df_tmp["sheet_name"] = str(sh_name)
            frames.append(df_tmp)
        if not frames:
            raise ValueError("엑셀 파일에 유효한 시트가 없습니다.")
        df_l2 = pd.concat(frames, axis=0, ignore_index=True)
    except ImportError:
        st.error("엑셀 파일을 읽기 위해 'openpyxl'이 필요합니다.\n가상환경에서 다음을 실행해주세요:\n\npip install openpyxl")
        st.stop()
    except FileNotFoundError:
        st.error(f"소분류 예측 파일을 찾을 수 없습니다: '{sec_path}' 경로를 확인해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"소분류 예측 파일 로드 중 오류가 발생했습니다: {e}")
        st.stop()

    # 1) 날짜 파싱: time(YYYY-MM) 또는 date 허용
    if "date" in df_l2.columns:
        try:
            df_l2["date"] = pd.to_datetime(df_l2["date"], errors="coerce")
        except Exception:
            df_l2["date"] = pd.to_datetime(df_l2["date"].astype(str), errors="coerce")
    elif "time" in df_l2.columns:
        df_l2["date"] = pd.to_datetime(df_l2["time"], format="%Y-%m", errors="coerce")
    else:
        # 날짜 컬럼이 없으면 생성 시도(무조건 실패 대비)
        if "연월" in df_l2.columns:
            df_l2["date"] = pd.to_datetime(df_l2["연월"].astype(str), format="%Y%m", errors="coerce")
        else:
            st.error("소분류 파일에는 'time'(YYYY-MM) 또는 'date' 컬럼이 필요합니다.")
            st.stop()

    # 2) 컬럼 정규화: category_l1, category_l2, y_true, y_pred가 없으면 wide→long 시도
    req_cols_l2 = {"category_l1", "category_l2", "y_true", "y_pred"}
    if not req_cols_l2.issubset(df_l2.columns):
        # 후보 예측 컬럼 패턴
        pred_cols = [c for c in df_l2.columns if c.endswith("_pred_MA3") or c.endswith("_pred")]
        # 실제값 컬럼 후보
        ytrue_col = "y_true" if "y_true" in df_l2.columns else ("CCSI" if "CCSI" in df_l2.columns else None)

        if (ytrue_col is not None) and len(pred_cols) > 0:
            # category_l1/l2 후보 식별
            l1_col = "category_l1" if "category_l1" in df_l2.columns else None
            l2_col = "category_l2" if "category_l2" in df_l2.columns else None

            # melt 이후에도 시트 정보를 유지하기 위해 sheet_name을 id_vars에 포함
            id_core = ["time", "date", ytrue_col, l1_col, l2_col, "sheet_name"]
            id_vars = [c for c in id_core if c and c in df_l2.columns]

            # 🔹 이미 한 쌍의 실제/예측 컬럼이 존재하면 melt를 생략하고 표준 컬럼명으로 정규화
            # 실제값 후보
            actual_candidates = [col for col in ["y_true", "y", "actual", "CCSI"] if col in df_l2.columns]
            # 예측값 후보
            pred_candidates = [col for col in ["y_pred", "pred", "yhat"] if col in df_l2.columns]

            if actual_candidates and pred_candidates:
                a_col = actual_candidates[0]
                p_col = pred_candidates[0]
                if "y_true" not in df_l2.columns and a_col != "y_true":
                    df_l2 = df_l2.rename(columns={a_col: "y_true"})
                if "y_pred" not in df_l2.columns and p_col != "y_pred":
                    df_l2 = df_l2.rename(columns={p_col: "y_pred"})
                # 여기서는 wide→long 불필요하므로 그대로 진행
                pass
            else:
                # value_name 충돌 방지
                val_name = "y_pred"
                if val_name in df_l2.columns:
                    val_name = "__y_pred_melt__"

                df_l2 = df_l2.melt(
                    id_vars=id_vars,
                    value_vars=pred_cols,
                    var_name="__pred_col__",
                    value_name=val_name
                )

                # "__y_pred_melt__"로 생성된 경우 다시 y_pred로 표준화
                if val_name != "y_pred":
                    df_l2 = df_l2.rename(columns={val_name: "y_pred"})

                # 컬럼명 표준화
                df_l2 = df_l2.rename(columns={ytrue_col: "y_true"})
                # 대분류/소분류가 없었다면 예측 컬럼명에서 유추할 수 있도록 기본값 설정
                if "category_l1" not in df_l2.columns:
                    df_l2["category_l1"] = "대분류"
                if "category_l2" not in df_l2.columns:
                    # 예: "음식_커피_pred_MA3" → l1="음식", l2="커피" 식으로 분해 시도
                    parts = df_l2["__pred_col__"].str.replace("_pred_MA3","",regex=False)\
                                                 .str.replace("_pred","",regex=False)\
                                                 .str.split("_", n=1, expand=True)
                    if isinstance(parts, pd.DataFrame) and parts.shape[1] == 2:
                        df_l2["category_l1"] = parts[0]
                        df_l2["category_l2"] = parts[1]
                    else:
                        df_l2["category_l2"] = df_l2["__pred_col__"].str.replace("_pred_MA3","",regex=False)\
                                                                    .str.replace("_pred","",regex=False)
                # 보조 컬럼 제거
                if "__pred_col__" in df_l2.columns:
                    df_l2 = df_l2.drop(columns=["__pred_col__"])
        else:
            st.error("소분류 파일에 (category_l1, category_l2, y_true, y_pred) 또는 WIDE 형식('*_pred*')이 필요합니다.")
            st.stop()

    # 시트명이 카테고리 정보인 경우 보완: 비어있거나 없는 category_l1/l2 채우기
    if "sheet_name" in df_l2.columns:
        if "category_l1" not in df_l2.columns:
            df_l2["category_l1"] = df_l2["sheet_name"]
        else:
            df_l2["category_l1"] = df_l2["category_l1"].fillna(df_l2["sheet_name"])
        if "category_l2" not in df_l2.columns:
            df_l2["category_l2"] = df_l2["sheet_name"]
        else:
            df_l2["category_l2"] = df_l2["category_l2"].fillna(df_l2["sheet_name"])
        # 시트명이 "대분류>소분류" 형태라면 분해 시도
        parts_sheet = df_l2["sheet_name"].str.split(">", n=1, expand=True)
        if isinstance(parts_sheet, pd.DataFrame) and parts_sheet.shape[1] == 2:
            df_l2["category_l1"] = df_l2["category_l1"].fillna(parts_sheet[0])
            df_l2["category_l2"] = df_l2["category_l2"].fillna(parts_sheet[1])
        # sheet_name은 이후 선택 UI에 활용하므로 삭제하지 않음

    # 3) 기간 필터 적용 (사이드바)
    try:
        start_d = pd.to_datetime(date_start)
        end_d = pd.to_datetime(date_end)
    except Exception:
        start_d = df_l2["date"].min()
        end_d = df_l2["date"].max()
    m_l2 = (df_l2["date"] >= start_d) & (df_l2["date"] <= end_d)
    df_l2 = df_l2.loc[m_l2].copy()
    # CCSI는 전체 구간 유지, 예측값은 2024-01 이후만 표시
    cutoff = pd.to_datetime("2024-01-01")
    df_l2.loc[df_l2["date"] < cutoff, "y_pred"] = np.nan

    # 4) 상단: 전체 평균 평가지표 (필터 후 전체)
    overall_rmse = _rmse(df_l2["y_true"], df_l2["y_pred"])
    overall_mae  = _mae(df_l2["y_true"], df_l2["y_pred"])
    overall_mape = _mape(df_l2["y_true"], df_l2["y_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 RMSE", f"{overall_rmse:.2f}")
    c2.metric("전체 MAE", f"{overall_mae:.2f}")
    c3.metric("전체 MAPE(%)", f"{overall_mape:.1f}%")

    st.divider()

    # 5) 선택바
    use_sheet = "sheet_name" in df_l2.columns
    if use_sheet:
        # 시트명에서 _ 이후 접미어(RMSE 등) 제거, 'A>B'를 'A > B'로 표기
        sheet_list = sorted(df_l2["sheet_name"].dropna().unique().tolist())
        if not sheet_list:
            st.warning("시트(sheet_name) 목록이 비어 있습니다.")
            st.stop()
        def clean_sheet_label(sheet):
            s = str(sheet).strip()
            # 1) '_' 이후 접미어(RMSE 등) 제거
            if "_" in s:
                s = s.split("_", 1)[0].strip()
            # 2) '대분류>소분류' 형태를 보기 좋게
            if ">" in s:
                a, b = s.split(">", 1)
                return f"{a.strip()} > {b.strip()}"
            return s
        sheet_labels = [clean_sheet_label(s) for s in sheet_list]
        label_to_sheet = dict(zip(sheet_labels, sheet_list))
        pick_label = st.selectbox("소분류 선택", sheet_labels)
        pick_sheet = label_to_sheet[pick_label]
        sub_l2 = df_l2[df_l2["sheet_name"] == pick_sheet].copy()
        # 제목 표기를 위해 정제된 label 사용
        clean_title = clean_sheet_label(pick_sheet)
        if ">" in clean_title:
            a, b = clean_title.split(">", 1)
            disp_l1, disp_l2 = a.strip(), b.strip()
        else:
            disp_l1, disp_l2 = "", clean_title.strip()
        display_title = f"[{disp_l1} > {disp_l2}]" if disp_l1 else f"[{disp_l2}]"
    else:
        # 대분류 → 소분류 선택
        l1_list = sorted(df_l2["category_l1"].dropna().unique().tolist())
        if not l1_list:
            st.warning("대분류(category_l1) 값이 없습니다.")
            st.stop()
        pick_l1 = st.selectbox("대분류 선택", l1_list)
        l2_list = sorted(df_l2.loc[df_l2["category_l1"]==pick_l1, "category_l2"].dropna().unique().tolist())
        if not l2_list:
            st.warning(f"'{pick_l1}'에 소분류(category_l2)가 없습니다.")
            st.stop()
        pick_l2 = st.selectbox("소분류 선택", l2_list)
        sub_l2 = df_l2[(df_l2["category_l1"]==pick_l1) & (df_l2["category_l2"]==pick_l2)].copy()
        display_title = f"[{pick_l1} > {pick_l2}]"

    # 6) 선택 조합 성능 지표 (간단 버전, RMSE/MAE/MAPE만)
    l2_rmse = _rmse(sub_l2["y_true"], sub_l2["y_pred"])
    l2_mae  = _mae(sub_l2["y_true"], sub_l2["y_pred"])
    l2_mape = _mape(sub_l2["y_true"], sub_l2["y_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{l2_rmse:.2f}")
    c2.metric("MAE", f"{l2_mae:.2f}")
    c3.metric("MAPE(%)", f"{l2_mape:.1f}%")

    # 7) 라인 차트 (실제 vs 예측) — 항상 표시 (SHOW_L2_CHARTS 플래그 제거)
    sub_l2_agg = (
        sub_l2.groupby("date", as_index=False)
              .agg(y_true=("y_true","mean"), y_pred=("y_pred","mean"))
              .sort_values("date")
    )
    plot_df_l2 = sub_l2_agg.rename(columns={"y_true":"Actual", "y_pred":"Pred"})[["date","Actual","Pred"]]
    plot_df_l2_m = plot_df_l2.melt("date", var_name="series", value_name="value")

    chart_title = f"{display_title} Actual vs Pred (소분류)"
    if HAS_PLOTLY:
        fig_l2 = px.line(plot_df_l2_m, x="date", y="value", color="series", title=chart_title)
        fig_l2.update_traces(mode="lines+markers")
        fig_l2.for_each_trace(
            lambda tr: tr.update(line=dict(color="blue")) if tr.name == "Actual" else tr.update(line=dict(color="red"))
        )
        merged_l2 = sub_l2_agg.rename(columns={"y_true":"Actual","y_pred":"Pred"}).copy()
        merged_l2["error"] = merged_l2["Pred"] - merged_l2["Actual"]
        fig_l2.for_each_trace(
            lambda tr: tr.update(
                customdata=merged_l2[["error"]].values,
                hovertemplate=("실제:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            ) if tr.name == "Actual" else tr.update(
                customdata=merged_l2[["error"]].values,
                hovertemplate=("예측:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            )
        )
        fig_l2.update_xaxes(dtick="M1", tickformat="%Y-%m")
        st.plotly_chart(fig_l2, use_container_width=True)
    else:
        line_chart(plot_df_l2_m, x="date", y="value", color="series", title=chart_title)

    st.divider()

    # 8) (선택) 대분류 내 소분류들 성능 비교 표
    if "sheet_name" in df_l2.columns:
        # sheet_name: 표에서 대분류>소분류 형태라면 표기 정제
        rows_l2 = []
        for g, df_g in df_l2.groupby("sheet_name"):
            # 표에 label도 정제해서 보여주기
            if ">" in g:
                parts = g.split(">", 1)
                label = f"{parts[0].strip()} > {parts[1].strip()}"
            elif "_" in g:
                parts = g.split("_", 1)
                label = f"{parts[0].strip()} > {parts[1].strip()}"
            else:
                label = g.strip()
            rows_l2.append({
                "분류": label,
                "RMSE": _rmse(df_g["y_true"], df_g["y_pred"]),
                "MAE":  _mae(df_g["y_true"], df_g["y_pred"]),
                "MAPE(%)": _mape(df_g["y_true"], df_g["y_pred"]),
            })
        comp_l2 = pd.DataFrame(rows_l2).sort_values("RMSE")
        st.markdown("**소분류별 성능 비교**")
        st.dataframe(comp_l2, use_container_width=True)
    else:
        rows_l2 = []
        for g, df_g in df_l2[df_l2["category_l1"]==pick_l1].groupby("category_l2"):
            rows_l2.append({
                "category_l2": g,
                "RMSE": _rmse(df_g["y_true"], df_g["y_pred"]),
                "MAE":  _mae(df_g["y_true"], df_g["y_pred"]),
                "MAPE(%)": _mape(df_g["y_true"], df_g["y_pred"]),
            })
        comp_l2 = pd.DataFrame(rows_l2).sort_values("RMSE")
        st.markdown(f"**[{pick_l1}] 소분류별 성능 비교**")
        st.dataframe(comp_l2, use_container_width=True)
with tabs[5]:
    st.subheader("Step 5: 성능 비교 (Total vs 대분류 vs 소분류)")

    cutoff = pd.to_datetime("2024-01-01")
    # 공통 기간 필터
    try:
        start_d = pd.to_datetime(date_start)
        end_d = pd.to_datetime(date_end)
    except Exception:
        start_d = pd.to_datetime("1900-01-01")
        end_d = pd.to_datetime("2100-01-01")

    # ----------------------------
    # 1) Total (단일 CCSI 예측)
    # ----------------------------
    total_ok = False
    try:
        df_total = pd.read_csv(results_path)
        if {"time","y_true","y_pred"}.issubset(df_total.columns):
            df_total["date"] = pd.to_datetime(df_total["time"], format="%Y-%m", errors="coerce")
            df_total = df_total.sort_values("date")
            # 기간 필터
            df_total = df_total[(df_total["date"] >= start_d) & (df_total["date"] <= end_d)].copy()
            # 예측은 2024-01 이후만
            df_total.loc[df_total["date"] < cutoff, "y_pred"] = np.nan
            total_rmse = _rmse(df_total["y_true"], df_total["y_pred"])
            total_mae  = _mae(df_total["y_true"], df_total["y_pred"])
            total_mape = _mape(df_total["y_true"], df_total["y_pred"])
            total_ok = True
        else:
            st.warning("Total 결과 파일(results/ccsi_total2.csv)의 형식이 올바르지 않습니다.")
    except Exception as e:
        st.warning(f"Total 결과 로드 실패: {e}")
        total_rmse = total_mae = total_mape = np.nan

    # ----------------------------
    # 2) 대분류 (firstgrade)
    # ----------------------------
    l1_ok = False
    try:
        df_l1_cmp = pd.read_csv("results/ccsi_firstgrade.csv")
        # 날짜
        if "date" in df_l1_cmp.columns:
            df_l1_cmp["date"] = pd.to_datetime(df_l1_cmp["date"], errors="coerce")
        elif "time" in df_l1_cmp.columns:
            df_l1_cmp["date"] = pd.to_datetime(df_l1_cmp["time"], format="%Y-%m", errors="coerce")
        else:
            raise ValueError("대분류 파일에 'time' 또는 'date' 컬럼이 필요합니다.")
        # 형식 표준화 (wide → long 변환)
        req_cols_l1 = {"category_l1","y_true","y_pred"}
        if not req_cols_l1.issubset(df_l1_cmp.columns):
            pred_cols = [c for c in df_l1_cmp.columns if c.endswith("_pred_MA3") or c.endswith("_pred")]
            if ("CCSI" in df_l1_cmp.columns) and len(pred_cols) > 0:
                id_vars = [c for c in ["time","date","CCSI"] if c in df_l1_cmp.columns]
                df_l1_cmp = df_l1_cmp.melt(id_vars=id_vars, value_vars=pred_cols,
                                           var_name="category_l1", value_name="y_pred")
                df_l1_cmp = df_l1_cmp.rename(columns={"CCSI":"y_true"})
                df_l1_cmp["category_l1"] = (
                    df_l1_cmp["category_l1"]
                    .str.replace("_pred_MA3","",regex=False)
                    .str.replace("_pred","",regex=False)
                )
            else:
                raise ValueError("대분류 파일 포맷이 예상과 다릅니다.")
        # 기간 필터
        df_l1_cmp = df_l1_cmp[(df_l1_cmp["date"] >= start_d) & (df_l1_cmp["date"] <= end_d)].copy()
        # 예측 마스킹
        df_l1_cmp.loc[df_l1_cmp["date"] < cutoff, "y_pred"] = np.nan
        l1_rmse = _rmse(df_l1_cmp["y_true"], df_l1_cmp["y_pred"])
        l1_mae  = _mae(df_l1_cmp["y_true"], df_l1_cmp["y_pred"])
        l1_mape = _mape(df_l1_cmp["y_true"], df_l1_cmp["y_pred"])
        l1_ok = True
    except Exception as e:
        st.warning(f"대분류 결과 로드 실패: {e}")
        l1_rmse = l1_mae = l1_mape = np.nan

    # ----------------------------
    # 3) 소분류 (secgrade)
    # ----------------------------
    l2_ok = False
    try:
        sec_path = Path("results") / "ccsi_secgrade.xlsx"
        xls_all = pd.read_excel(sec_path, sheet_name=None)
        frames = []
        for sh, df_sh in xls_all.items():
            if df_sh is None or len(df_sh)==0:
                continue
            df_t = df_sh.copy()
            # 날짜 파싱
            if "date" in df_t.columns:
                df_t["date"] = pd.to_datetime(df_t["date"], errors="coerce")
            elif "time" in df_t.columns:
                df_t["date"] = pd.to_datetime(df_t["time"], format="%Y-%m", errors="coerce")
            else:
                if "연월" in df_t.columns:
                    df_t["date"] = pd.to_datetime(df_t["연월"].astype(str), format="%Y%m", errors="coerce")
                else:
                    continue
            # 실제/예측 표준화
            if "y_true" not in df_t.columns:
                if "CCSI" in df_t.columns:
                    df_t = df_t.rename(columns={"CCSI":"y_true"})
            if "y_pred" not in df_t.columns:
                pred_cols = [c for c in df_t.columns if c.endswith("_pred_MA3") or c.endswith("_pred")]
                if len(pred_cols)==1:
                    df_t = df_t.rename(columns={pred_cols[0]:"y_pred"})
                elif len(pred_cols)>1:
                    # 여러 예측컬럼이면 우선 첫 번째 사용 (간단비교 목적)
                    df_t = df_t.rename(columns={pred_cols[0]:"y_pred"})
                else:
                    continue
            df_t["sheet_name"] = str(sh)
            frames.append(df_t[["date","y_true","y_pred","sheet_name"]])
        if not frames:
            raise ValueError("소분류 시트에서 유효한 데이터가 없습니다.")
        df_l2_cmp = pd.concat(frames, ignore_index=True)
        # 기간 필터
        df_l2_cmp = df_l2_cmp[(df_l2_cmp["date"] >= start_d) & (df_l2_cmp["date"] <= end_d)].copy()
        # 예측 마스킹
        df_l2_cmp.loc[df_l2_cmp["date"] < cutoff, "y_pred"] = np.nan
        l2_rmse = _rmse(df_l2_cmp["y_true"], df_l2_cmp["y_pred"])
        l2_mae  = _mae(df_l2_cmp["y_true"], df_l2_cmp["y_pred"])
        l2_mape = _mape(df_l2_cmp["y_true"], df_l2_cmp["y_pred"])
        l2_ok = True
    except Exception as e:
        st.warning(f"소분류 결과 로드 실패: {e}")
        l2_rmse = l2_mae = l2_mape = np.nan

    # ----------------------------
    # 4) 비교 표 + 시각화
    # ----------------------------
    comp = pd.DataFrame({
        "Level": ["Total","대분류","소분류"],
        "RMSE":  [total_rmse, l1_rmse, l2_rmse],
        "MAE":   [total_mae,  l1_mae,  l2_mae],
        "MAPE":  [total_mape, l1_mape, l2_mape],
    })

    c1, c2 = st.columns([2,1])
    with c1:
        if HAS_PLOTLY:
            import plotly.express as px
            comp_m = comp.melt(id_vars="Level", value_vars=["RMSE","MAE","MAPE"], var_name="Metric", value_name="Value")
            fig_cmp = px.bar(comp_m, x="Level", y="Value", color="Metric", barmode="group",
                             title="Total vs 대분류 vs 소분류 성능 비교")
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.bar_chart(comp.set_index("Level")[["RMSE","MAE","MAPE"]])

    with c2:
        st.markdown("**요약 지표**")
        st.dataframe(comp.style.format({"RMSE":"{:.2f}","MAE":"{:.2f}","MAPE":"{:.1f}"}), use_container_width=True)
with tabs[6]:
    st.subheader("Final Insight")
    st.markdown("""
    **요약**
    - **예측 반영 구간**: 실제 CCSI는 전체 기간을, 예측은 **2024-01 이후만** 반영했습니다.
    - **분해 효과**: 단일 예측보다 **대분류/소분류 분해 후 합산**이 전반적으로 오차를 줄이는 경향을 보였습니다.  

    **권장 해석 순서**
    1) **Step 1**에서 전체 실제 vs 예측 추세와 잔차를 확인  
    2) **Step 3**에서 9개 **대분류** 성능을 비교해 민감도가 큰 업종을 식별  
    3) **Step 4**에서 해당 대분류의 **소분류**로 드릴다운해 패턴을 점검  
    4) **Step 5**에서 **Total vs 대분류 vs 소분류**의 지표를 한 번에 비교

    **향후 개선 포인트**
    - **외생 변수**(금리·물가·고용) 보강 및 시차 최적화
    - **명절/정책/프로모션** 더미와 이상치 처리 고도화
    - 소분류별 **가중 합산 전략**(구성비 동적 추정) 도입
    - 최신 월에 대한 **예측 캘리브레이션**(최근성 가중/선형 보정)

    """)