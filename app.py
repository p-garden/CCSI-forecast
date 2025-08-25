# app.py
# 실행: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
st.caption("파일 업로드 없이 Step 1에서 즉시 인터랙티브 데모 그래프가 나타납니다.")

# --------------------------
# 사이드바 (분석 설정)
# --------------------------
st.sidebar.header("⚙️ 분석 설정")
date_start = st.sidebar.date_input("시작일", value=datetime(2022,1,1))
date_end   = st.sidebar.date_input("종료일", value=datetime(2025,6,1))

# 기본값 설정 (CSV 결과 사용 모드)
use_results = True
results_path = "results/ccsi_prediction_vs_actual.csv"

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
    st.caption("의미: 업종 k의 소비액은 **과거 CCSI**, **Lag/Window 피처**, **계절성**, **외생 변수**를 입력으로 하는 모델 \(f_k\)로 **설명/예측**됩니다.")

    st.markdown("### 평가 지표")
    c1,c2,c3 = st.columns(3)
    c1.metric("권장 지표", "RMSE")
    c2.metric("보조 지표", "MAE")
    c3.metric("상대 오차", "MAPE (%)")
    

# --------------------------
# Step 3: 소분류 예측
# --------------------------
with tabs[3]:
    st.subheader("소분류별 소비액 예측 (대분류 → 소분류)")
    cat_path2 = st.text_input("소분류 CSV 경로 (같은 파일 가능)", value="data/consumption_by_category.csv", key="cat2")
    try:
        cat2_df = _load_category_csv(cat_path2)
    except Exception as e:
        st.warning(f"소비액 CSV 로드 실패: {e}")
        st.stop()

    req_cols2 = {"category_l1", "category_l2", "amount", "amount_pred"}
    if not req_cols2.issubset(cat2_df.columns):
        st.error("필수 컬럼이 없습니다. (category_l1, category_l2, amount, amount_pred)")
        st.stop()

    # 기간 필터
    m2 = (cat2_df["date"] >= pd.to_datetime(date_start)) & (cat2_df["date"] <= pd.to_datetime(date_end))
    cat2_v = cat2_df.loc[m2].copy()

    # 대분류 선택 → 소분류 선택
    l1_opts = sorted(cat2_v["category_l1"].dropna().unique().tolist())
    if not l1_opts:
        st.warning("선택 가능한 대분류가 없습니다.")
        st.stop()
    l1_pick = st.selectbox("대분류 선택", l1_opts)
    sub_l1 = cat2_v[cat2_v["category_l1"] == l1_pick].copy()

    l2_opts = sorted(sub_l1["category_l2"].dropna().unique().tolist())
    if not l2_opts:
        st.warning("선택 가능한 소분류가 없습니다.")
        st.stop()
    l2_pick = st.selectbox("소분류 선택", l2_opts)

    sub2 = sub_l1[sub_l1["category_l2"] == l2_pick].copy()

    # 지표
    rmse2 = _rmse(sub2["amount"], sub2["amount_pred"])
    mae2  = _mae(sub2["amount"], sub2["amount_pred"])
    mape2 = _mape(sub2["amount"], sub2["amount_pred"])

    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse2:.2f}")
    c2.metric("MAE", f"{mae2:.2f}")
    c3.metric("MAPE(%)", f"{mape2:.1f}%")

    # 라인 차트 (Hover: 실제/예측/오차/날짜)
    plot_df2 = sub2.rename(columns={"amount":"Actual", "amount_pred":"Pred"})[["date","Actual","Pred"]]
    plot_df2_m = plot_df2.melt("date", var_name="series", value_name="value")

    if HAS_PLOTLY:
        import plotly.express as px
        fig2 = px.line(plot_df2_m, x="date", y="value", color="series", title=f"[{l1_pick} / {l2_pick}] Actual vs Pred (소분류)")
        fig2.update_traces(mode="lines+markers")
        merged2 = sub2[["date","amount","amount_pred"]].copy()
        merged2["error"] = merged2["amount_pred"] - merged2["amount"]
        fig2.for_each_trace(
            lambda tr: tr.update(
                customdata=merged2[["error"]].values,
                hovertemplate=("실제:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            ) if tr.name == "Actual" else tr.update(
                customdata=merged2[["error"]].values,
                hovertemplate=("예측:<br>%{y:.2f}<br>오차: %{customdata[0]:.2f}<br>날짜: %{x|%Y-%m}")
            )
        )
        fig2.update_xaxes(dtick="M1", tickformat="%Y-%m")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        line_chart(plot_df2_m, x="date", y="value", color="series", title=f"[{l1_pick} / {l2_pick}] Actual vs Pred (소분류)")

    st.divider()
    st.markdown(f"**[{l1_pick}] 소분류별 성능 비교**")
    rows2 = []
    for g, df_g in sub_l1.groupby("category_l2"):
        rows2.append({
            "category_l2": g,
            "RMSE": _rmse(df_g["amount"], df_g["amount_pred"]),
            "MAE":  _mae(df_g["amount"], df_g["amount_pred"]),
            "MAPE(%)": _mape(df_g["amount"], df_g["amount_pred"]),
        })
    comp2 = pd.DataFrame(rows2).sort_values("RMSE")
    st.dataframe(comp2, use_container_width=True)

# --------------------------
# 나머지 탭: 데이터 연결 전까지 안내만
# --------------------------
with tabs[4]:
    st.subheader("소분류 Drill-down & 예측")
    st.write("대분류 선택 → 소분류 상세 예측을 표시합니다. (데이터 연결 후 활성화)")
with tabs[5]:
    st.subheader("성능 비교")
    st.write("CCSI 단일 예측 vs (대분류/소분류) 예측 합산 성능 비교 표/그래프. (데이터 연결 후 활성화)")
with tabs[6]:
    st.subheader("결론 및 확장성")
    st.markdown(
        """
        **핵심 결론**  
        - 단일 CCSI 예측은 변동성과 구성요인 복잡성 때문에 오차가 큼  
        - **소비액을 대분류→소분류로 세분화**하여 직접 예측하고 합산하면 설명력과 예측력이 개선  

        **확장 제안**  
        - 소분류별 리드-랙(lead-lag) 분석으로 선행성 탐색  
        - 물가/금리/실업률 등 외생 변수 결합  
        - 특수 이벤트(명절, 정책 등) 마커링 & 이상치 감지
        """
    )