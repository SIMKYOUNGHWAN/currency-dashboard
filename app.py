import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# 2. 금융 데이터 로드 (yf.download 일괄 수집으로 안정성 확보)
@st.cache_data(ttl=3600)
def load_market_data():
    ticker_map = {
        'USDKRW': 'KRW=X',
        'USDGEL': 'GEL=X',
        'TNX': '^TNX',     # 미 10년물 국채 금리
        'VIX': '^VIX',     # 변동성 지수
        'Oil': 'CL=F',      # WTI 원유 선물
        'SPX': '^GSPC'     # S&P 500 지수
    }
    
    symbols = list(ticker_map.values())
    
    # 여러 티커를 한 번에 수집하여 차원/날짜 정렬 오류 방지
    raw_data = yf.download(symbols, period="3y", progress=False)['Close']
    
    # 컬럼 이름을 한글/알기 쉬운 명칭 매핑용 영문으로 변경
    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)
    
    # 앞/뒤 결측치 보완
    df = df.ffill().bfill()
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
    
    # 모델에 필요한 컬럼만 추출 후 결측치 제거
    df_model = df[features + ['Target']].dropna()
    
    X = df_model[features]
    y = df_model['Target']
    
    n_samples = len(X)
    feature_labels = ['10년물 금리 변동률', 'VIX 수준', '유가 변동률', 'S&P500 변동률']
    
    # 샘플 수가 부족할 경우 예외 방지 안전 장치
    if n_samples < 10:
        return 0.5, 0.50, pd.Series([0.25]*4, index=feature_labels)
        
    # 샘플 수에 맞게 n_splits 동적 조절 (최대 3개, 최소 2개)
    n_splits = min(3, max(2, n_samples // 30))
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, model.predict(X_te)))
        
    # 전체 데이터로 최종 모델 학습
    model.fit(X, y)
    latest_x = X.iloc[[-1]]
    
    prob_up = model.predict_proba(latest_x)[0][1]
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
c2.metric("원화 상승(상승률) 확률", f"{krw_prob*100:.1f}%", delta=f"검증 정확도 {krw_acc*100:.1f}%")

c3.metric("현재 USDGEL", f"{curr_gel:,.4f} GEL")
c4.metric("라리화 상승(상승률) 확률", f"{gel_prob*100:.1f}%", delta=f"검증 정확도 {gel_acc*100:.1f}%")

st.markdown("---")

# 매크로 지표 독립 다중 Y축 그래프 시각화
st.subheader("📊 주요 매크로 지표 추이 (독립 Y축 그래프)")

fig, ax1 = plt.subplots(figsize=(12, 5))

# 축 1: TNX (금리)
color1 = '#1f77b4'
ax1.set_xlabel('날짜')
ax1.set_ylabel('미 10년물 국채금리 (%)', color=color1)
ax1.plot(df.index[-120:], df['TNX'].iloc[-120:], color=color1, label='TNX (%)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# 축 2: VIX (변동성)
ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('변동성지수 (VIX)', color=color2)
ax2.plot(df.index[-120:], df['VIX'].iloc[-120:], color=color2, label='VIX', linewidth=1.5, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

# 축 3: Oil (원유)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))
color3 = '#2ca02c'
ax3.set_ylabel('WTI 원유 ($/bbl)', color=color3)
ax3.plot(df.index[-120:], df['Oil'].iloc[-120:], color=color3, label='WTI Oil ($)', linewidth=1.5, linestyle=':')
ax3.tick_params(axis='y', labelcolor=color3)

plt.title("최근 6개월 주요 매크로 변수 동향", fontsize=13, pad=10)
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

st.caption("데이터 출처: Yahoo Finance | 매시간 자동으로 최신 시장 데이터를 수집하여 업데이트합니다.")
