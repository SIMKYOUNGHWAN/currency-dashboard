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

st.title("🌐 거시경제 지표 기반 환율 예측 대시보드")
st.caption("미국 달러 대 원화(USDKRW) 및 조지아 라리화(USDGEL - NBG 공식 연동) 4주 추세 분석 모델")

# 데이터 캐시 비우기 버튼
if st.sidebar.button("🔄 데이터 캐시 강제 리셋"):
    st.cache_data.clear()
    st.rerun()


# 3. 보조지표 계산 함수 (RSI) - 상단에 정의
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# 4. 조지아 중앙은행(NBG) 단일 날짜 환율 수집 함수
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


# 5. NBG 데이터를 병렬 수집하여 타겟 시계열 인덱스에 맞추는 함수
# [버그 수정 #1] 기존 코드는 3일 간격으로 샘플링한 뒤 interpolate(method='time')로
# 빈 구간을 "선형 직선"으로 채웠다. 이 인위적인 직선 구간에서는 등락(delta)이
# 계속 한쪽 방향으로만 미세하게 움직이는 것처럼 계산되어, RSI가 몇 달간 0(또는 100)
# 근처에 눌러붙는 왜곡이 발생한다(실제 대시보드 캡처에서 GEL RSI가 4~8월 내내
# 바닥에 붙어있다가 9월에 급등한 원인). 코드1(main.py)이 실제로 쓰는 방식대로
# ffill/bfill만 사용해 "실제 마지막 고시값이 다음 갱신 전까지 유지"되는 계단식
# 형태로 채운다. 이렇게 해야 RSI/변동성 같은 모멘텀 지표가 실제 환율 움직임을
# 반영하게 된다.
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
        s_filled = (
            s.reindex(full_idx)
            .ffill()   # [수정] interpolate(method='time') 대신 ffill/bfill만 사용
            .bfill()
            .reindex(target_index)
        )
        return s_filled

    try:
        url = "https://open.er-api.com/v6/latest/USD"
        res = requests.get(url, timeout=5).json()
        base_rate = res.get("rates", {}).get("GEL", 2.70)
    except Exception:
        base_rate = 2.70
    return pd.Series(base_rate, index=target_index)


# 6. 전체 마켓 데이터 로드
@st.cache_data(ttl=600)
def load_market_data():
    ticker_map = {
        'USDKRW': 'KRW=X',
        'TNX': '^TNX',     # 미 10년물 국채 금리
        'VIX': '^VIX',     # 변동성 지수
        'Oil': 'CL=F',     # WTI 원유 선물
        'SPX': '^GSPC',    # S&P 500 지수
        'DXY': 'DX-Y.NYB'  # 달러 인덱스
    }

    symbols = list(ticker_map.values())
    raw_data = yf.download(symbols, period="3y", progress=False)['Close']

    inv_map = {v: k for k, v in ticker_map.items()}
    df = raw_data.rename(columns=inv_map)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # [버그 수정 #2] 기존 코드는 이 아래에서 df['USDGEL']을 먼저 채운 뒤,
    # 그 다음 줄의 df.interpolate(...)가 "전체 컬럼"에 다시 걸리면서
    # 어렵게 ffill로 채운 GEL 값에 선형보간이 재차 섞여 들어갔다.
    # -> 매크로/증시 데이터(주말 결측)에 대한 interpolate를 먼저 끝내고,
    #    GEL은 그 이후에 별도 로직으로 채워서 다시 덮어써지지 않게 한다.
    df = df.interpolate(method='time', limit_direction='both').ffill().bfill()
    df['USDGEL'] = fetch_usdgel_series(df.index)

    return df


# 7. AI 모델 학습 및 예측 함수
# [버그 수정 #3, 가장 중요] 기존 코드는 feature_cols + 'Target'을 한 번에 dropna()했다.
# Target = shift(-20)이라 최근 20거래일은 Target이 NaN이라 통째로 삭제되고,
# 그 결과 latest_x = X.iloc[[-1]]이 "진짜 최신 데이터"가 아니라 약 1개월 전 데이터를
# 가리키게 된다. KPI 카드의 "현재 환율"은 진짜 최신 값인데, 그 옆의 AI 예측 확률은
# 한 달 전 스냅샷 기준이었던 것 -> 코드1(main.py)의 패턴대로 "feature 계산 가능 여부"
# 기준(df_clean)과 "학습용(Target 존재)" 기준(train_df)을 분리해서
# 최신 예측은 반드시 진짜 마지막 날짜의 feature를 쓰도록 고친다.
#
# 추가로, 기존 코드는 KRW/GEL 두 통화에 완전히 동일한 매크로 변수 5개만 재사용해서
# 두 통화의 변수중요도 차트가 사실상 똑같이 나왔다. 코드1처럼 통화 자신의
# 수익률/이동평균비율/RSI/변동성을 feature에 추가해 통화별 특성을 반영한다.
def train_and_predict(data, target_symbol):
    df = data.copy()
    horizon = 20  # 20 거래일 ≈ 4주

    # --- 통화 자기 자신의 기술적 지표 (코드1의 analyze_currency_enhanced 방식) ---
    df[f'{target_symbol}_Ret_4W'] = df[target_symbol].pct_change(horizon)
    df[f'{target_symbol}_MA12_Ratio'] = df[target_symbol] / df[target_symbol].rolling(60).mean() - 1
    df[f'{target_symbol}_MA24_Ratio'] = df[target_symbol] / df[target_symbol].rolling(120).mean() - 1
    df[f'{target_symbol}_RSI14'] = calculate_rsi(df[target_symbol], period=14)
    df[f'{target_symbol}_Vol12W'] = df[target_symbol].pct_change().rolling(60).std()

    # --- 공통 매크로 지표 ---
    df['TNX_Ret_4W'] = df['TNX'].pct_change(horizon)
    df['TNX_Level'] = df['TNX']
    df['VIX_Level'] = df['VIX']
    df['VIX_Change_4W'] = df['VIX'].pct_change(horizon)
    df['Oil_Ret_4W'] = df['Oil'].pct_change(horizon)
    df['SPX_Ret_4W'] = df['SPX'].pct_change(horizon)
    df['DXY_Ret_4W'] = df['DXY'].pct_change(horizon)

    df['Target'] = (df[target_symbol].shift(-horizon) > df[target_symbol]).astype(int)

    feature_cols = [
        f'{target_symbol}_Ret_4W',
        f'{target_symbol}_MA12_Ratio',
        f'{target_symbol}_MA24_Ratio',
        f'{target_symbol}_RSI14',
        f'{target_symbol}_Vol12W',
        'TNX_Ret_4W', 'TNX_Level',
        'VIX_Level', 'VIX_Change_4W',
        'Oil_Ret_4W', 'SPX_Ret_4W', 'DXY_Ret_4W',
    ]

    # feature 계산이 가능한 구간(=가장 최근 날짜까지 포함)
    df_clean = df.dropna(subset=feature_cols)
    # 학습에는 정답(Target)이 존재하는 구간만 사용 (최근 horizon일은 자연히 제외됨)
    train_df = df_clean.dropna(subset=['Target'])

    n_samples = len(train_df)

    if n_samples < 30 or len(np.unique(train_df['Target'])) < 2 or len(df_clean) == 0:
        fallback = pd.Series([1.0 / len(feature_cols)] * len(feature_cols), index=feature_cols)
        return 0.5, 0.50, fallback

    X = train_df[feature_cols]
    y = train_df['Target']

    n_splits = min(5, max(2, n_samples // 60))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(np.unique(y_tr)) >= 2:
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model.predict(X_te)))

    model.fit(X, y)

    # [수정] 반드시 df_clean(진짜 최신 날짜)에서 마지막 행을 뽑는다.
    latest_x = df_clean[feature_cols].iloc[[-1]]

    classes = list(model.classes_)
    if 1 in classes:
        idx_1 = classes.index(1)
        prob_up = model.predict_proba(latest_x)[0][idx_1]
    else:
        prob_up = 0.0

    accuracy = np.mean(scores) if scores else 0.50
    feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return prob_up, accuracy, feature_imp


# 8. 메인 데이터 처리 및 리포트 화면 구현
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

# [수정] 최근 120거래일(~6개월)만 잘라서 보여주던 것을 코드1(main.py)처럼
# 수집한 3년치 전체 데이터로 확장. yfinance/NBG 모두 period="3y"로 이미
# 3년치를 받아오고 있었으므로, 여기서 슬라이싱만 없애면 된다.
recent_df = df.copy()
recent_df['KRW_MA20'] = recent_df['USDKRW'].rolling(20).mean()
recent_df['KRW_MA60'] = recent_df['USDKRW'].rolling(60).mean()
recent_df['GEL_MA20'] = recent_df['USDGEL'].rolling(20).mean()
recent_df['GEL_MA60'] = recent_df['USDGEL'].rolling(60).mean()

# 1) 타겟 환율 추이 그래프
st.subheader("📈 타겟 환율 추이 및 이동평균선 (최근 3년)")
col_fx1, col_fx2 = st.columns(2)

with col_fx1:
    fig_krw, ax_krw = plt.subplots(figsize=(6, 3))
    ax_krw.plot(recent_df.index, recent_df['USDKRW'], color='black', linewidth=1.8, label='USDKRW')
    ax_krw.plot(recent_df.index, recent_df['KRW_MA20'], color='orange', linestyle='--', linewidth=1.2, label='MA20')
    ax_krw.plot(recent_df.index, recent_df['KRW_MA60'], color='red', linestyle=':', linewidth=1.2, label='MA60')
    ax_krw.set_title("USDKRW (KRW/USD)", fontsize=11, pad=8)
    ax_krw.grid(True, linestyle='--', alpha=0.5)
    ax_krw.legend(loc='upper left', fontsize=8)
    fig_krw.autofmt_xdate(rotation=30)
    fig_krw.tight_layout()
    st.pyplot(fig_krw)

with col_fx2:
    fig_gel, ax_gel = plt.subplots(figsize=(6, 3))
    ax_gel.plot(recent_df.index, recent_df['USDGEL'], color='navy', linewidth=1.8, label='USDGEL')
    ax_gel.plot(recent_df.index, recent_df['GEL_MA20'], color='orange', linestyle='--', linewidth=1.2, label='MA20')
    ax_gel.plot(recent_df.index, recent_df['GEL_MA60'], color='red', linestyle=':', linewidth=1.2, label='MA60')
    ax_gel.set_title("USDGEL (National Bank of Georgia)", fontsize=11, pad=8)
    ax_gel.grid(True, linestyle='--', alpha=0.5)
    ax_gel.legend(loc='upper left', fontsize=8)
    fig_gel.autofmt_xdate(rotation=30)
    fig_gel.tight_layout()
    st.pyplot(fig_gel)

st.markdown("---")

# 2) 주요 매크로 지표 추이
st.subheader("📊 주요 매크로 지표 추이 (독립 Y축 그래프)")
st.caption("🟦 **파란색 (좌측 Y축)**: 미 10년물 국채 금리 (TNX) | 🟧 **주황색 점선 (우측1 Y축)**: VIX 변동성 지수 | 🟩 **초록색 점선 (우측2 Y축)**: WTI 원유 가격 ($)")

fig, ax1 = plt.subplots(figsize=(12, 4))

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
ax1.grid(True, linestyle='--', alpha=0.3)
plt.title("Macro Trends (TNX, VIX, WTI Oil)", fontsize=12, pad=10)
fig.tight_layout()

st.pyplot(fig)

st.markdown("---")

# 3) 달러 인덱스 & 미국 증시 추이
st.subheader("💵 달러 인덱스 (DXY) 및 미국 증시 (S&P 500) 추이")
st.caption("🟪 **보라색 (좌측 Y축)**: 달러 인덱스 (DXY) | 🌲 **진초록색 점선 (우측 Y축)**: S&P 500 지수")

fig_dxy, ax_dxy = plt.subplots(figsize=(12, 4))
ax_spx = ax_dxy.twinx()

color_dxy = 'purple'
color_spx = 'darkgreen'

line_dxy = ax_dxy.plot(recent_df.index, recent_df['DXY'], color=color_dxy, label='Dollar Index (DXY)', linewidth=1.8)
line_spx = ax_spx.plot(recent_df.index, recent_df['SPX'], color=color_spx, linestyle='--', label='S&P 500 Index', linewidth=1.5)

ax_dxy.set_ylabel('DXY Index', color=color_dxy)
ax_spx.set_ylabel('S&P 500 Index', color=color_spx)
ax_dxy.tick_params(axis='y', labelcolor=color_dxy)
ax_spx.tick_params(axis='y', labelcolor=color_spx)

lines_macro = line_dxy + line_spx
labels_macro = [l.get_label() for l in lines_macro]
ax_dxy.legend(lines_macro, labels_macro, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
ax_dxy.grid(True, linestyle='--', alpha=0.3)
fig_dxy.tight_layout()

st.pyplot(fig_dxy)

st.markdown("---")

# 4) 기술적 보조지표 (RSI & 연율화 변동성)
st.subheader("📉 기술적 보조지표 분석 (RSI & 연율화 변동성)")
col_rsi, col_vol = st.columns(2)

recent_df['KRW_RSI'] = calculate_rsi(recent_df['USDKRW'], 14)
recent_df['GEL_RSI'] = calculate_rsi(recent_df['USDGEL'], 14)

recent_df['KRW_Vol'] = recent_df['USDKRW'].pct_change().rolling(20).std() * np.sqrt(252) * 100
recent_df['GEL_Vol'] = recent_df['USDGEL'].pct_change().rolling(20).std() * np.sqrt(252) * 100

with col_rsi:
    fig_rsi, ax_rsi = plt.subplots(figsize=(6, 3))
    ax_rsi.plot(recent_df.index, recent_df['KRW_RSI'], color='black', label='KRW RSI(14)')
    ax_rsi.plot(recent_df.index, recent_df['GEL_RSI'], color='navy', label='GEL RSI(14)')
    ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)')
    ax_rsi.axhline(30, color='blue', linestyle='--', alpha=0.6, label='Oversold (30)')
    ax_rsi.set_title("RSI (Overbought / Oversold)", fontsize=11, pad=8)
    ax_rsi.grid(True, linestyle='--', alpha=0.3)
    ax_rsi.legend(loc='upper left', fontsize=8)
    fig_rsi.autofmt_xdate(rotation=30)
    fig_rsi.tight_layout()
    st.pyplot(fig_rsi)

with col_vol:
    fig_vol, ax_vol = plt.subplots(figsize=(6, 3))
    ax_vol.plot(recent_df.index, recent_df['KRW_Vol'], color='teal', label='KRW Volatility')
    ax_vol.plot(recent_df.index, recent_df['GEL_Vol'], color='darkslateblue', label='GEL Volatility')
    ax_vol.set_title("20-Day Annualized Volatility (%)", fontsize=11, pad=8)
    ax_vol.grid(True, linestyle='--', alpha=0.3)
    ax_vol.legend(loc='upper left', fontsize=8)
    fig_vol.autofmt_xdate(rotation=30)
    fig_vol.tight_layout()
    st.pyplot(fig_vol)

st.markdown("---")

# 5) 변수 중요도 분석
# [수정] feature_cols가 통화별로 12개로 늘어났으므로, 코드1처럼 Top 6만 표시해 가독성 유지
st.subheader("🔍 변수 중요도 분석 (Feature Importance, Top 6)")
col_a, col_b = st.columns(2)

with col_a:
    st.write("**USDKRW 영향 변수**")
    st.bar_chart(krw_imp.head(6))

with col_b:
    st.write("**USDGEL 영향 변수**")
    st.bar_chart(gel_imp.head(6))

st.caption("데이터 출처: Yahoo Finance & National Bank of Georgia (NBG) | 매시간 자동으로 최신 시장 데이터를 수집하여 업데이트합니다.")
