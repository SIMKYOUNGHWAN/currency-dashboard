import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Streamlit Cloud 서버 튕김 방지 (최상단 필수 배치)
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# NBG API SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 웹 페이지 기본 레이아웃 설정
st.set_page_config(
    page_title="거시경제 환율 예측 대시보드",
    page_icon="📈",
    layout="wide"
)

# 2. Streamlit 상단 툴바 및 헤더 숨김 CSS
st.markdown("""
    <style>
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 거시경제 지표 기반 환율 예측 대시보드 (고도화 모델)")
st.caption("미국 달러 대 원화(USDKRW) 및 조지아 라리화(USDGEL - NBG 연동) AI 분석 및 실시간 시뮬레이터")

# 3. 사이드바 - 설정 및 시나리오 시뮬레이터 (What-If)
st.sidebar.header("🎛️ 예측 설정 및 시뮬레이션")

horizon_weeks = st.sidebar.selectbox(
    "⏱️ 예측 타임프레임 선택",
    options=[1, 2, 4],
    index=2,
    format_func=lambda x: f"{x}주일 후 ({x*5}영업일)"
)
horizon_days = horizon_weeks * 5

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 실시간 시나리오 시뮬레이터")
st.sidebar.caption("거시경제 변수가 급변할 때 AI 예측 확률의 변화를 실시간으로 테스트합니다.")

oil_shift = st.sidebar.slider("WTI 원유 가격 변동 충격 (%)", -20.0, 20.0, 0.0, step=1.0) / 100.0
tnx_shift = st.sidebar.slider("미 10년물 금리 변동 충격 (%)", -20.0, 20.0, 0.0, step=1.0) / 100.0
dxy_shift = st.sidebar.slider("달러 인덱스(DXY) 변동 충격 (%)", -10.0, 10.0, 0.0, step=0.5) / 100.0

if st.sidebar.button("🔄 데이터 캐시 강제 리셋"):
    st.cache_data.clear()
    st.rerun()

# 4. 보조지표 계산 함수 (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# 5. 조지아 중앙은행(NBG) 수집 함수
def fetch_nbg_single_date(dt):
    date_str = dt.strftime("%Y-%m-%d")
    url = f"https://nbg.gov.ge/gw/api/ct/monetarypolicy/currencies/en/json/?date={date_str}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        res = requests.get(url, headers=headers, timeout=5, verify=False)
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                for curr in data[0].get("currencies", []):
                    if curr.get("code") == "USD":
                        return dt, float(curr["rate"])
    except Exception:
        pass
    return dt, None

def fetch_usdgel_series(target_index):
    if len(target_index) == 0:
        return pd.Series(dtype=float)
    start_date = target_index.min()
    end_date = target_index.max()
    sample_dates = pd.date_range(start=start_date, end=end_date, freq="3D")
    records = {}
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = executor.map(fetch_nbg_single_date, sample_dates)
        for dt, rate in results:
            if rate is not None:
                records[dt] = rate
    if records:
        s = pd.Series(records).sort_index()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        full_idx = s.index.union(target_index)
        return (s.reindex(full_idx)
                .interpolate(method="time")
                .reindex(target_index)
                .ffill()
                .bfill())
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        res = requests.get(url, timeout=5).json()
        base_rate = res.get("rates", {}).get("GEL", 2.70)
    except Exception:
        base_rate = 2.70
    return pd.Series(base_rate, index=target_index)

# 6. 마켓 데이터 수집 (위안화 USDCNY 포함)
@st.cache_data(ttl=600)
def load_market_data():
    ticker_map = {
        'USDKRW': 'KRW=X',
        'TNX': '^TNX',     # 미 10년물 국채 금리
        'VIX': '^VIX',     # 변동성 지수
        'Oil': 'CL=F',     # WTI 원유 선물
        'SPX': '^GSPC',    # S&P 500 지수
        'DXY': 'DX-Y.NYB', # 달러 인덱스
        'SOX': '^SOX',     # 필라델피아 반도체 지수
        'USDCNY': 'CNY=X'  # 중국 위안화
    }
    symbols = list(ticker_map.values())
    raw_data = yf.download(symbols, period="3y", progress=False)['Close']
    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['USDGEL'] = fetch_usdgel_series(df.index)
    df = df.interpolate(method='time', limit_direction='both').ffill().bfill()
    return df

# 7. AI 모델 학습 및 시나리오 시뮬레이션 연동 함수
def train_and_predict(data, target_symbol, horizon_days, oil_s=0.0, tnx_s=0.0, dxy_s=0.0):
    df_temp = data.copy()
    
    # Target 환율 기술적 지표 피처
    df_temp['Target_Ret_1W'] = df_temp[target_symbol].pct_change(5)
    df_temp['Target_Ret_H'] = df_temp[target_symbol].pct_change(horizon_days)
    df_temp['Target_MA_Ratio'] = df_temp[target_symbol] / df_temp[target_symbol].rolling(20).mean()
    df_temp['Target_RSI'] = calculate_rsi(df_temp[target_symbol], 14)
    
    # 거시경제 및 무역 여건, 위안화 피처
    df_temp['TNX_Ret_H'] = df_temp['TNX'].pct_change(horizon_days)
    df_temp['VIX_Level'] = df_temp['VIX']
    df_temp['Oil_Ret_H'] = df_temp['Oil'].pct_change(horizon_days)
    df_temp['SPX_Ret_H'] = df_temp['SPX'].pct_change(horizon_days)
    df_temp['DXY_Ret_H'] = df_temp['DXY'].pct_change(horizon_days)
    df_temp['CNY_Ret_H'] = df_temp['USDCNY'].pct_change(horizon_days)
    df_temp['Trade_Proxy_Ret'] = (df_temp['SOX'] / df_temp['Oil']).pct_change(horizon_days)
    
    # 선택한 타임프레임 후 상승 여부 Target
    df_temp['Target'] = (df_temp[target_symbol].shift(-horizon_days) > df_temp[target_symbol]).astype(int)
    
    features = [
        'Target_Ret_1W', 'Target_Ret_H', 'Target_MA_Ratio', 'Target_RSI',
        'TNX_Ret_H', 'VIX_Level', 'Oil_Ret_H', 'SPX_Ret_H', 'DXY_Ret_H', 'CNY_Ret_H', 'Trade_Proxy_Ret'
    ]
    feature_labels = [
        '1W Return', f'{horizon_weeks}W Return', '20D MA Ratio', 'RSI Index',
        '10Y Yield', 'VIX Index', 'WTI Oil', 'S&P 500', 'Dollar Index', 'USD/CNY', 'Trade Proxy'
    ]
    
    df_model = df_temp[features + ['Target']].dropna()
    X = df_model[features]
    y = df_model['Target']
    
    n_samples = len(X)
    if n_samples < 20 or len(np.unique(y)) < 2:
        return 0.5, 0.50, pd.Series([0.1]*len(features), index=feature_labels)
        
    n_splits = min(4, max(2, n_samples // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 과적합 방지 모델 규제
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3, 
        min_samples_leaf=5, 
        random_state=42
    )
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(np.unique(y_tr)) >= 2:
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model.predict(X_te)))
            
    model.fit(X, y)
    
    # 최신 데이터에 시나리오 충격 적용
    latest_x = X.iloc[[-1]].copy()
    latest_x['Oil_Ret_H'] += oil_s
    latest_x['TNX_Ret_H'] += tnx_s
    latest_x['DXY_Ret_H'] += dxy_s
    if oil_s != 0.0:
        latest_x['Trade_Proxy_Ret'] -= oil_s * 0.5
        
    classes = list(model.classes_)
    if 1 in classes:
        idx_1 = classes.index(1)
        prob_up = model.predict_proba(latest_x)[0][idx_1]
    else:
        prob_up = 0.0
        
    accuracy = np.mean(scores) if scores else 0.50
    feature_imp = pd.Series(model.feature_importances_, index=feature_labels)
    
    return prob_up, accuracy, feature_imp

# 8. 메인 데이터 처리 및 화면 구현
with st.spinner("마켓 및 NBG 데이터를 수집하여 AI 모델을 재학습 중입니다..."):
    df = load_market_data()
    krw_prob, krw_acc, krw_imp = train_and_predict(df, 'USDKRW', horizon_days, oil_shift, tnx_shift, dxy_shift)
    gel_prob, gel_acc, gel_imp = train_and_predict(df, 'USDGEL', horizon_days, oil_shift, tnx_shift, dxy_shift)

# 파생 지표 전체 3년 계산
df['KRW_MA20'] = df['USDKRW'].rolling(20).mean()
df['KRW_MA60'] = df['USDKRW'].rolling(60).mean()
df['GEL_MA20'] = df['USDGEL'].rolling(20).mean()
df['GEL_MA60'] = df['USDGEL'].rolling(60).mean()
df['Trade_Proxy'] = df['SOX'] / df['Oil']

df['KRW_RSI'] = calculate_rsi(df['USDKRW'], 14)
df['GEL_RSI'] = calculate_rsi(df['USDGEL'], 14)
df['KRW_Vol'] = df['USDKRW'].pct_change().rolling(20).std() * np.sqrt(252) * 100
df['GEL_Vol'] = df['USDGEL'].pct_change().rolling(20).std() * np.sqrt(252) * 100

# 시나리오 설정 시 안내 표시
if oil_shift != 0.0 or tnx_shift != 0.0 or dxy_shift != 0.0:
    st.warning(f"⚠️ **실시간 시나리오 적용 중**: WTI({oil_shift*100:+.1f}%), 금리({tnx_shift*100:+.1f}%), DXY({dxy_shift*100:+.1f}%) 충격을 반영한 예측 결과입니다.")

# KPI 요약 카드
st.subheader(f"📌 {horizon_weeks}주 후 환율 방향성 AI 예측 확률")
c1, c2, c3, c4 = st.columns(4)

curr_krw = df['USDKRW'].iloc[-1]
curr_gel = df['USDGEL'].iloc[-1]

c1.metric("현재 USDKRW", f"{curr_krw:,.2f} 원")
c2.metric(f"USDKRW ({horizon_weeks}주 후) 상승 확률", f"{krw_prob*100:.1f}%", delta=f"검증 정확도 {krw_acc*100:.1f}%")

c3.metric("현재 USDGEL (NBG)", f"{curr_gel:,.4f} GEL")
c4.metric(f"USDGEL ({horizon_weeks}주 후) 상승 확률", f"{gel_prob*100:.1f}%", delta=f"검증 정확도 {gel_acc*100:.1f}%")

st.markdown("---")

# 1) 타겟 환율 추이 그래프 (최근 3년)
st.subheader("📈 타겟 환율 추이 및 이동평균선 (최근 3년)")
col_fx1, col_fx2 = st.columns(2)

with col_fx1:
    fig_krw, ax_krw = plt.subplots(figsize=(6, 3))
    ax_krw.plot(df.index, df['USDKRW'], color='black', linewidth=1.5, label='USDKRW')
    ax_krw.plot(df.index, df['KRW_MA20'], color='orange', linestyle='--', linewidth=1.1, label='MA20')
    ax_krw.plot(df.index, df['KRW_MA60'], color='red', linestyle=':', linewidth=1.1, label='MA60')
    ax_krw.set_title("USDKRW (KRW/USD)", fontsize=11, pad=8)
    ax_krw.grid(True, linestyle='--', alpha=0.5)
    ax_krw.legend(loc='upper left', fontsize=8)
    fig_krw.autofmt_xdate(rotation=30)
    fig_krw.tight_layout()
    st.pyplot(fig_krw)

with col_fx2:
    fig_gel, ax_gel = plt.subplots(figsize=(6, 3))
    ax_gel.plot(df.index, df['USDGEL'], color='navy', linewidth=1.5, label='USDGEL')
    ax_gel.plot(df.index, df['GEL_MA20'], color='orange', linestyle='--', linewidth=1.1, label='MA20')
    ax_gel.plot(df.index, df['GEL_MA60'], color='red', linestyle=':', linewidth=1.1, label='MA60')
    ax_gel.set_title("USDGEL (National Bank of Georgia)", fontsize=11, pad=8)
    ax_gel.grid(True, linestyle='--', alpha=0.5)
    ax_gel.legend(loc='upper left', fontsize=8)
    fig_gel.autofmt_xdate(rotation=30)
    fig_gel.tight_layout()
    st.pyplot(fig_gel)

st.markdown("---")

# 2) 무역 여건 & 위안화 동향 그래프 (시작일=100 기준 정규화)
st.subheader("🚢 한국 무역 여건(Trade Proxy) 및 위안화(USDCNY) 커플링")
st.caption("💡 단위가 서로 다른 3개 지표의 상대 변동률을 왜곡 없이 비교하기 위해 **시작일 기준(=100)**으로 정규화한 차트입니다.")

# 3개 지표 시작일(100) 기준 정규화 지수 생성
df_norm = pd.DataFrame(index=df.index)
df_norm['USDKRW'] = (df['USDKRW'] / df['USDKRW'].iloc[0]) * 100
df_norm['Trade_Proxy'] = (df['Trade_Proxy'] / df['Trade_Proxy'].iloc[0]) * 100
df_norm['USDCNY'] = (df['USDCNY'] / df['USDCNY'].iloc[0]) * 100

fig_trade, ax_t = plt.subplots(figsize=(12, 4))

ax_t.plot(df_norm.index, df_norm['USDKRW'], color='black', label='USDKRW (원/달러)', linewidth=1.5)
ax_t.plot(df_norm.index, df_norm['Trade_Proxy'], color='dodgerblue', linestyle='--', label='Trade Proxy (무역여건: SOX/Oil)', linewidth=1.3)
ax_t.plot(df_norm.index, df_norm['USDCNY'], color='crimson', linestyle='-', label='USDCNY (위안화)', linewidth=1.4)

ax_t.set_ylabel('상대 변동 지수 (시작일 = 100)', fontsize=10)
ax_t.grid(True, linestyle='--', alpha=0.3)
ax_t.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.title("Normalized Trend Comparison (USDKRW vs Trade Proxy vs USDCNY)", fontsize=12, pad=10)
fig_trade.tight_layout()

st.pyplot(fig_trade)
st.info("💡 **차트 해석**: 정규화(Base=100)를 거치면서 원/달러(검은선)와 위안화(빨간선)의 동조화(커플링) 현상 및 무역 여건(파란 점선)과의 역상관 관계를 왜곡 없이 한눈에 비교할 수 있습니다.")

st.markdown("---")

# 3) 주요 매크로 지표 추이 (최근 3년)
st.subheader("📊 주요 매크로 지표 추이 (최근 3년)")
fig, ax1 = plt.subplots(figsize=(12, 4))

color1 = '#1f77b4'
ax1.set_xlabel('Date')
ax1.set_ylabel('US 10Y Treasury (%)', color=color1)
line1 = ax1.plot(df.index, df['TNX'], color=color1, label='US 10Y Yield (TNX)', linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('VIX Index', color=color2)
line2 = ax2.plot(df.index, df['VIX'], color=color2, label='VIX Index', linewidth=1.2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))
color3 = '#2ca02c'
ax3.set_ylabel('WTI Oil ($/bbl)', color=color3)
line3 = ax3.plot(df.index, df['Oil'], color=color3, label='WTI Oil ($)', linewidth=1.2, linestyle=':')
ax3.tick_params(axis='y', labelcolor=color3)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.3)
plt.title("Macro Trends (TNX, VIX, WTI Oil)", fontsize=12, pad=10)
fig.tight_layout()

st.pyplot(fig)

st.markdown("---")

# 4) 기술적 보조지표 및 변수 중요도
st.subheader("🔍 변수 중요도 분석 (Feature Importance)")
col_a, col_b = st.columns(2)

with col_a:
    st.write(f"**USDKRW ({horizon_weeks}주 후 예측) 영향 변수**")
    st.bar_chart(krw_imp)

with col_b:
    st.write(f"**USDGEL ({horizon_weeks}주 후 예측) 영향 변수**")
    st.bar_chart(gel_imp)

st.caption("데이터 출처: Yahoo Finance & National Bank of Georgia (NBG) | 실시간 시장 데이터를 반영하여 동적으로 재학습됩니다.")
