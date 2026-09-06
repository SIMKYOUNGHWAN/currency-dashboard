import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# 1. 웹 페이지 기본 레이아웃 설정
st.set_page_config(
    page_title="거시경제 환율 예측 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("🌐 거시경제 지표 기반 환율 예측 대시보드")
st.caption("미국 달러 대 원화(USDKRW) 및 라리화(USDGEL) 4주 추세 분석 모델")

# 데이터 캐시 비우기 버튼
if st.sidebar.button("🔄 데이터 캐시 강제 리셋"):
    st.cache_data.clear()
    st.rerun()


# 2. 라리화(USDGEL) 전용 고신뢰도 수집 함수
# [수정] 실데이터인지 시뮬레이션(대체) 데이터인지 여부를 함께 반환하도록 변경
def fetch_usdgel_series():
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        res = requests.get(url, timeout=5).json()
        base_rate = res.get("rates", {}).get("GEL", 2.65)
    except Exception:
        base_rate = 2.65

    try:
        df_gel = yf.Ticker("GEL=X").history(period="3y")['Close']
        if len(df_gel) > 50 and df_gel.std() > 0.001:
            df_gel.index = pd.to_datetime(df_gel.index).tz_localize(None)
            return df_gel, False  # 실데이터, 시뮬레이션 아님
    except Exception:
        pass

    # 실데이터 수집 실패 시: 시뮬레이션 데이터로 대체하되, 반드시 플래그로 알림
    dates = pd.date_range(end=pd.Timestamp.now(), periods=750, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0, 0.003, size=len(dates))
    price_path = base_rate * np.exp(np.cumsum(returns))
    return pd.Series(price_path, index=dates, name='USDGEL'), True  # 시뮬레이션 데이터임을 명시


# 3. 전체 마켓 데이터 로드
@st.cache_data(ttl=600)
def load_market_data():
    ticker_map = {
        'USDKRW': 'KRW=X',
        'TNX': '^TNX',     # 미 10년물 국채 금리
        'VIX': '^VIX',     # 변동성 지수
        'Oil': 'CL=F',      # WTI 원유 선물
        'SPX': '^GSPC'     # S&P 500 지수
    }

    symbols = list(ticker_map.values())

    # [수정] 다중 티커 다운로드 실패/부분실패에 대한 예외처리 추가
    try:
        raw_data = yf.download(symbols, period="3y", progress=False)['Close']
    except Exception as e:
        st.error(f"시장 데이터 다운로드 중 오류가 발생했습니다: {e}")
        raw_data = pd.DataFrame(columns=symbols)

    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)

    # 혹시 일부 종목이 통째로 빠졌을 경우 빈 컬럼이라도 만들어 이후 로직이 죽지 않도록 함
    for col in inv_map.values():
        if col not in df.columns:
            df[col] = np.nan

    df.index = pd.to_datetime(df.index).tz_localize(None)

    # GEL 데이터 결합
    gel_series, is_simulated = fetch_usdgel_series()
    df['USDGEL'] = gel_series

    # 결측치 보완 (선형 보간 및 앞뒤 채우기)
    df = df.interpolate(method='time', limit_direction='both').ffill().bfill()
    return df, is_simulated


# 4. AI 모델 학습 및 예측 함수
# [수정] "오늘 기준" 예측용 피처와 "학습용" 피처/타겟을 분리
#        기존 코드는 Target이 NaN인 최근 20영업일 행을 dropna로 제거한 뒤
#        그 데이터프레임에서 latest_x = X.iloc[[-1]] 을 뽑았기 때문에,
#        실제로는 "오늘"이 아니라 "약 4주 전" 시점의 지표로 예측하는 버그가 있었음.
def train_and_predict(data, target_symbol):
    df = data.copy()

    df['TNX_Ret_4W'] = df['TNX'].pct_change(20)
    df['VIX_Level'] = df['VIX']
    df['Oil_Ret_4W'] = df['Oil'].pct_change(20)
    df['SPX_Ret_4W'] = df['SPX'].pct_change(20)

    df['Target'] = (df[target_symbol].shift(-20) > df[target_symbol]).astype(int)

    features = ['TNX_Ret_4W', 'VIX_Level', 'Oil_Ret_4W', 'SPX_Ret_4W']
    feature_labels = ['10Y Yield', 'VIX Index', 'WTI Oil', 'S&P 500']

    # --- (A) 오늘 예측에 쓸 피처: Target 유무와 무관하게 "피처만" 존재하면 됨 ---
    df_features_only = df[features].dropna()
    if df_features_only.empty:
        return 0.5, 0.50, pd.Series([0.25] * 4, index=feature_labels)
    latest_x = df_features_only.iloc[[-1]]  # 진짜 오늘(가장 최근) 시점의 피처

    # --- (B) 모델 학습용 데이터: Target까지 존재해야 하므로 최근 20일은 자연스럽게 제외됨 ---
    df_train = df[features + ['Target']].dropna()
    X = df_train[features]
    y = df_train['Target']

    n_samples = len(X)

    if n_samples < 10 or len(np.unique(y)) < 2:
        return 0.5, 0.50, pd.Series([0.25] * 4, index=feature_labels)

    n_splits = min(3, max(2, n_samples // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if len(np.unique(y_tr)) >= 2:
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model.predict(X_te)))

    model.fit(X, y)

    classes = list(model.classes_)
    if 1 in classes:
        idx_1 = classes.index(1)
        prob_up = model.predict_proba(latest_x)[0][idx_1]
    else:
        prob_up = 0.0

    accuracy = np.mean(scores) if scores else 0.50
    feature_imp = pd.Series(model.feature_importances_, index=feature_labels)

    return prob_up, accuracy, feature_imp


# 5. 화면 출력 부분
with st.spinner("최신 마켓 데이터 수집 및 예측 모델 실행 중..."):
    df, gel_is_simulated = load_market_data()
    krw_prob, krw_acc, krw_imp = train_and_predict(df, 'USDKRW')
    gel_prob, gel_acc, gel_imp = train_and_predict(df, 'USDGEL')

# [수정] 시뮬레이션 데이터 사용 시 사용자에게 명확히 경고
if gel_is_simulated:
    st.warning(
        "⚠️ USDGEL(라리화) 실시간 데이터를 가져오지 못해, 현재 화면의 USDGEL 관련 수치는 "
        "**실제 시장 데이터가 아닌 임의로 생성된 시뮬레이션 데이터**입니다. "
        "해당 값을 실제 투자/의사결정 판단 근거로 사용하지 마세요."
    )

# KPI 요약 카드
st.subheader("📌 4주 후 환율 방향성 AI 예측 확률")
c1, c2, c3, c4 = st.columns(4)

curr_krw = df['USDKRW'].iloc[-1]
curr_gel = df['USDGEL'].iloc[-1]

c1.metric("현재 USDKRW", f"{curr_krw:,.2f} 원")
c2.metric("USDKRW(환율) 상승 확률", f"{krw_prob*100:.1f}%", delta=f"검증 정확도 {krw_acc*100:.1f}%")

gel_label = "현재 USDGEL" + (" (시뮬레이션)" if gel_is_simulated else "")
c3.metric(gel_label, f"{curr_gel:,.4f} GEL")
c4.metric("USDGEL(환율) 상승 확률", f"{gel_prob*100:.1f}%", delta=f"검증 정확도 {gel_acc*100:.1f}%")

st.markdown("---")

# 타겟 환율 추이 그래프
st.subheader("📈 타겟 환율 추이 (최근 6개월)")
col_fx1, col_fx2 = st.columns(2)

recent_df = df.iloc[-120:]  # 최근 6개월(120영업일)

with col_fx1:
    fig_krw, ax_krw = plt.subplots(figsize=(6, 3))
    ax_krw.plot(recent_df.index, recent_df['USDKRW'], color='#1f77b4', linewidth=1.8)
    ax_krw.set_title("USDKRW (KRW/USD)", fontsize=11, pad=8)
    ax_krw.grid(True, linestyle='--', alpha=0.5)

    valid_krw = recent_df['USDKRW'].dropna()
    if not valid_krw.empty:
        krw_min, krw_max = valid_krw.min(), valid_krw.max()
        if not np.isnan(krw_min) and not np.isnan(krw_max):
            ax_krw.set_ylim(krw_min * 0.98, krw_max * 1.02)

    fig_krw.autofmt_xdate(rotation=30)
    fig_krw.tight_layout()
    st.pyplot(fig_krw)

with col_fx2:
    gel_title = "USDGEL (GEL/USD)" + (" - 시뮬레이션" if gel_is_simulated else "")
    fig_gel, ax_gel = plt.subplots(figsize=(6, 3))
    ax_gel.plot(recent_df.index, recent_df['USDGEL'], color='#ff7f0e', linewidth=1.8)
    ax_gel.set_title(gel_title, fontsize=11, pad=8)
    ax_gel.grid(True, linestyle='--', alpha=0.5)

    valid_gel = recent_df['USDGEL'].dropna()
    if not valid_gel.empty:
        gel_min, gel_max = valid_gel.min(), valid_gel.max()
        if not np.isnan(gel_min) and not np.isnan(gel_max):
            if gel_min == gel_max:
                ax_gel.set_ylim(gel_min * 0.98, gel_max * 1.02)
            else:
                ax_gel.set_ylim(gel_min * 0.995, gel_max * 1.005)

    fig_gel.autofmt_xdate(rotation=30)
    fig_gel.tight_layout()
    st.pyplot(fig_gel)

st.markdown("---")

# 매크로 지표 독립 다중 Y축 그래프 시각화
st.subheader("📊 주요 매크로 지표 추이 (독립 Y축 그래프)")

fig, ax1 = plt.subplots(figsize=(12, 4.5))

color1 = '#1f77b4'
ax1.set_xlabel('Date')
ax1.set_ylabel('US 10Y Treasury (%)', color=color1)
ax1.plot(recent_df.index, recent_df['TNX'], color=color1, label='TNX (%)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('VIX Index', color=color2)
ax2.plot(recent_df.index, recent_df['VIX'], color=color2, label='VIX', linewidth=1.5, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))
color3 = '#2ca02c'
ax3.set_ylabel('WTI Oil ($/bbl)', color=color3)
ax3.plot(recent_df.index, recent_df['Oil'], color=color3, label='WTI Oil ($)', linewidth=1.5, linestyle=':')
ax3.tick_params(axis='y', labelcolor=color3)

plt.title("Recent 6-Month Macro Trends (TNX, VIX, WTI Oil)", fontsize=12, pad=10)
fig.tight_layout()

st.pyplot(fig)

# 주요 영향 변수 표시
st.subheader("🔍 변수 중요도 분석 (Feature Importance)")
col_a, col_b = st.columns(2)

with col_a:
    st.write("**USDKRW 영향 변수**")
    st.bar_chart(krw_imp)

with col_b:
    st.write("**USDGEL 영향 변수**" + (" (시뮬레이션 데이터 기반)" if gel_is_simulated else ""))
    st.bar_chart(gel_imp)

st.caption("데이터 출처: Yahoo Finance & Open ER API | 매시간 자동으로 최신 시장 데이터를 수집하여 업데이트합니다.")
