import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
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

# 라리화(USDGEL) 전용 고신뢰도 데이터 수집 함수 (Frankfurter Open FX API 백업 연동)
def fetch_usdgel_series():
    # 1차: 야후 파이낸스 데이터 시도 및 변동성 검증
    try:
        raw = yf.download('GEL=X', period="3y", progress=False)['Close']
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        s = raw.dropna()
        if len(s) > 100 and s.std() > 0.005:
            return s
    except Exception:
        pass

    # 2차: Frankfurter Open FX API 활용 (무료 실시간/과거 GEL 시계열 데이터)
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        url = f"https://api.frankfurter.app/{start_date}..{end_date}?from=USD&to=GEL"
        res = requests.get(url, timeout=5).json()
        if 'rates' in res:
            rates = {k: v['GEL'] for k, v in res['rates'].items() if 'GEL' in v}
            s = pd.Series(rates)
            s.index = pd.to_datetime(s.index)
            s.name = 'USDGEL'
            if len(s) > 50:
                return s
    except Exception:
        pass

    # 3차: 예비 티커 시도
    try:
        raw = yf.download('USDGEL=X', period="3y", progress=False)['Close']
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        return raw
    except Exception:
        return None

# 2. 전체 마켓 데이터 로드
@st.cache_data(ttl=3600)
def load_market_data():
    ticker_map = {
        'USDKRW': 'KRW=X',
        'TNX': '^TNX',     # 미 10년물 국채 금리
        'VIX': '^VIX',     # 변동성 지수
        'Oil': 'CL=F',      # WTI 원유 선물
        'SPX': '^GSPC'     # S&P 500 지수
    }
    
    symbols = list(ticker_map.values())
    raw_data = yf.download(symbols, period="3y", progress=False)['Close']
    
    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)
    
    # USDGEL 전용 데이터 병합
    gel_series = fetch_usdgel_series()
    if gel_series is not None:
        df['USDGEL'] = gel_series
    else:
        df['USDGEL'] = np.nan

    # 시간대(timezone) 정렬 및 결측치 보완
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.interpolate(method='time', limit_direction='both').ffill().bfill()
    return df

# 3. AI 모델 학습 및 예측 함수
def train_and_predict(data, target_symbol):
    df = data.copy()
    
    # 매크로 파생 변수 생성
    df['TNX_Ret_4W'] = df['TNX'].pct_change(20)
    df['VIX_Level'] = df['VIX']
    df['Oil_Ret_4W'] = df['Oil'].pct_change(20)
    df['SPX_Ret_4W'] = df['SPX'].pct_change(20)
    
    # 타겟 설정: 4주(20영업일) 후 상승 여부 (1: 상승, 0: 하락/보합)
    df['Target'] = (df[target_symbol].shift(-20) > df[target_symbol]).astype(int)
    
    features = ['TNX_Ret_4W', 'VIX_Level', 'Oil_Ret_4W', 'SPX_Ret_4W']
    df_model = df[features + ['Target']].dropna()
    
    X = df_model[features]
    y = df_model['Target']
    
    n_samples = len(X)
    feature_labels = ['10Y Yield', 'VIX Index', 'WTI Oil', 'S&P 500']
    
    if n_samples < 10 or len(np.unique(y)) < 2:
        return 0.5, 0.50, pd.Series([0.25]*4, index=feature_labels)
        
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
    latest_x = X.iloc[[-1]]
    
    classes = list(model.classes_)
    if 1 in classes:
        idx_1 = classes.index(1)
        prob_up = model.predict_proba(latest_x)[0][idx_1]
    else:
        prob_up = 0.0
        
    accuracy = np.mean(scores) if scores else 0.50
    feature_imp = pd.Series(model.feature_importances_, index=feature_labels)
    
    return prob_up, accuracy, feature_imp

# 4. 화면 출력 부분
with st.spinner("최신 마켓 데이터 수집 및 예측 모델 실행 중..."):
    df = load_market_data()
    krw_prob, krw_acc, krw_imp = train_and_predict(df, 'USDKRW')
    gel_prob, gel_acc, gel_imp = train_and_predict(df, 'USDGEL')

# KPI 요약 카드
st.subheader("📌 4주 후 환율 방향성 AI 예측 확률")
c1, c2, c3, c4 = st.columns(4)

curr_krw = df['USDKRW'].iloc[-1]
curr_gel = df['USDGEL'].iloc[-1]

c1.metric("현재 USDKRW", f"{curr_krw:,.2f} 원")
c2.metric("USDKRW(환율) 상승 확률", f"{krw_prob*100:.1f}%", delta=f"검증 정확도 {krw_acc*100:.1f}%")

c3.metric("현재 USDGEL", f"{curr_gel:,.4f} GEL")
c4.metric("USDGEL(환율) 상승 확률", f"{gel_prob*100:.1f}%", delta=f"검증 정확도 {gel_acc*100:.1f}%")

st.markdown("---")

# 동적 Y축 스케일링이 적용된 환율 추이 그래프
st.subheader("📈 타겟 환율 추이 (최근 6개월)")
col_fx1, col_fx2 = st.columns(2)

recent_df = df.iloc[-120:] # 최근 6개월(120영업일)

with col_fx1:
    fig_krw, ax_krw = plt.subplots(figsize=(6, 3))
    ax_krw.plot(recent_df.index, recent_df['USDKRW'], color='#1f77b4', linewidth=1.8)
    ax_krw.set_title("USDKRW (KRW/USD)", fontsize=11, pad=8)
    ax_krw.grid(True, linestyle='--', alpha=0.5)
    
    krw_min, krw_max = recent_df['USDKRW'].min(), recent_df['USDKRW'].max()
    ax_krw.set_ylim(krw_min * 0.98, krw_max * 1.02)
    fig_krw.autofmt_xdate(rotation=30)
    fig_krw.tight_layout()
    st.pyplot(fig_krw)

with col_fx2:
    fig_gel, ax_gel = plt.subplots(figsize=(6, 3))
    ax_gel.plot(recent_df.index, recent_df['USDGEL'], color='#ff7f0e', linewidth=1.8)
    ax_gel.set_title("USDGEL (GEL/USD)", fontsize=11, pad=8)
    ax_gel.grid(True, linestyle='--', alpha=0.5)
    
    gel_min, gel_max = recent_df['USDGEL'].min(), recent_df['USDGEL'].max()
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
    st.write("**USDGEL 영향 변수**")
    st.bar_chart(gel_imp)

st.caption("데이터 출처: Yahoo Finance & Frankfurter API | 매시간 자동으로 최신 시장 데이터를 수집하여 업데이트합니다.")
