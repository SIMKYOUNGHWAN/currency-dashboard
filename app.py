import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

st.title("🌐 거시경제 지표 기반 환율 예측 대시보드")
st.caption("미국 달러 대 원화(USDKRW) 및 조지아 라리화(USDGEL - NBG 공식 연동) 4주 추세 분석 모델")

# 데이터 캐시 비우기 버튼
if st.sidebar.button("🔄 데이터 캐시 강제 리셋"):
    st.cache_data.clear()
    st.rerun()

# 2. 조지아 중앙은행(NBG) 단일 날짜 환율 수집 함수
def fetch_nbg_single_date(dt):
    date_str = dt.strftime("%Y-%m-%d")
    url = f"https://nbg.gov.ge/gw/api/ct/monetarypolicy/currencies/en/json/?date={date_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
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

# 3. NBG 데이터를 병렬 수집하여 타겟 시계열 인덱스에 맞추는 함수
def fetch_usdgel_series(target_index):
    if len(target_index) == 0:
        return pd.Series(dtype=float)
        
    start_date = target_index.min()
    end_date = target_index.max()
    
    # 3일 간격 샘플링으로 빠른 병렬 데이터 수집
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
        
        # 타겟 인덱스 기준으로 시계열 보간(Interpolation) 및 정렬
        full_idx = s.index.union(target_index)
        s_interpolated = (
            s.reindex(full_idx)
            .interpolate(method="time")
            .reindex(target_index)
            .ffill()
            .bfill()
        )
        return s_interpolated

    # 비상 예외 시 실시간 기준 환율 API 활용
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        res = requests.get(url, timeout=5).json()
        base_rate = res.get("rates", {}).get("GEL", 2.70)
    except Exception:
        base_rate = 2.70
    return pd.Series(base_rate, index=target_index)

# 4. 전체 마켓 데이터 로드
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
    raw_data = yf.download(symbols, period="3y", progress=False)['Close']
    
    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # 조지아 중앙은행(NBG) 실제 라리화 환율 시계열 결합
    df['USDGEL'] = fetch_usdgel_series(df.index)
    
    # 결측치 최종 보완
    df = df.interpolate(method='time', limit_direction='both').ffill().bfill()
    return df

# 5. AI 모델 학습 및 예측 함수
def train_and_predict(data, target_symbol):
    df = data.copy()
    
    df['TNX_Ret_4W'] = df['TNX'].pct_change(20)
    df['VIX_Level'] = df['VIX']
    df['Oil_Ret_4W'] = df['Oil'].pct_change(20)
    df['SPX_Ret_4W'] = df['SPX'].pct_change(20)
    
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

# 6. 화면 출력
with st.spinner("조지아 중앙은행(NBG) 및 글로벌 마켓 데이터 수집 중..."):
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

c3.metric("현재 USDGEL (NBG)", f"{curr_gel:,.4f} GEL")
c4.metric("USDGEL(환율) 상승 확률", f"{gel_prob*100:.1f}%", delta=f"검증 정확도 {gel_acc*100:.1f}%")

st.markdown("---")

# 타겟 환율 추이 그래프
st.subheader("📈 타겟 환율 추이 (최근 6개월)")
col_fx1, col_fx2 = st.columns(2)

recent_df = df.iloc[-120:]

with col_fx1:
    fig_krw, ax_krw = plt.subplots(figsize=(6, 3))
    ax_krw.plot(recent_df.index, recent_df['USDKRW'], color='#1f77b4', linewidth=1.8)
    ax_krw.set_title("USDKRW (KRW/USD)", fontsize=11, pad=8)
    ax_krw.grid(True, linestyle='--', alpha=0.5)
    
    valid_krw = recent_df['USDKRW'].dropna()
    if not valid_krw.empty:
        ax_krw.set_ylim(valid_krw.min() * 0.98, valid_krw.max() * 1.02)
            
    fig_krw.autofmt_xdate(rotation=30)
    fig_krw.tight_layout()
    st.pyplot(fig_krw)

with col_fx2:
    fig_gel, ax_gel = plt.subplots(figsize=(6, 3))
    ax_gel.plot(recent_df.index, recent_df['USDGEL'], color='#ff7f0e', linewidth=1.8)
    ax_gel.set_title("USDGEL (National Bank of Georgia)", fontsize=11, pad=8)
    ax_gel.grid(True, linestyle='--', alpha=0.5)
    
    valid_gel = recent_df['USDGEL'].dropna()
    if not valid_gel.empty:
        g_min, g_max = valid_gel.min(), valid_gel.max()
        if g_min == g_max:
            ax_gel.set_ylim(g_min * 0.98, g_max * 1.02)
        else:
            ax_gel.set_ylim(g_min * 0.995, g_max * 1.005)
                
    fig_gel.autofmt_xdate(rotation=30)
    fig_gel.tight_layout()
    st.pyplot(fig_gel)

st.markdown("---")

# 주요 매크로 지표 추이
st.subheader("📊 주요 매크로 지표 추이 (독립 Y축 그래프)")
st.caption("🟦 **파란색 (좌측 Y축)**: 미 10년물 국채 금리 (TNX) | 🟧 **주황색 점선 (우측1 Y축)**: VIX 변동성 지수 | 🟩 **초록색 점선 (우측2 Y축)**: WTI 원유 가격 ($)")

fig, ax1 = plt.subplots(figsize=(12, 4.5))

color1 = '#1f77b4'
ax1.set_xlabel('Date')
ax1.set_ylabel('US 10Y Treasury (%)', color=color1)
line1 = ax1.plot(recent_df.index, recent_df['TNX'], color=color1, label='US 10Y Yield (TNX)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('VIX Index', color=color2)
line2 = ax2.plot(recent_df.index, recent_df['VIX'], color=color2, label='VIX Index', linewidth=1.5, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))
color3 = '#2ca02c'
ax3.set_ylabel('WTI Oil ($/bbl)', color=color3)
line3 = ax3.plot(recent_df.index, recent_df['Oil'], color=color3, label='WTI Oil ($)', linewidth=1.5, linestyle=':')
ax3.tick_params(axis='y', labelcolor=color3)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.title("Recent 6-Month Macro Trends (TNX, VIX, WTI Oil)", fontsize=12, pad=10)
fig.tight_layout()

st.pyplot(fig)

# 변수 중요도 분석
st.subheader("🔍 변수 중요도 분석 (Feature Importance)")
col_a, col_b = st.columns(2)

with col_a:
    st.write("**USDKRW 영향 변수**")
    st.bar_chart(krw_imp)

with col_b:
    st.write("**USDGEL 영향 변수**")
    st.bar_chart(gel_imp)

st.caption("데이터 출처: Yahoo Finance & National Bank of Georgia (NBG) | 매시간 자동으로 최신 시장 데이터를 수집하여 업데이트합니다.")
