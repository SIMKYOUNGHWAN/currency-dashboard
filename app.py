from concurrent.futures import ThreadPoolExecutor
import os
import platform
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# [사용자 설정] 텔레그램 정보 입력
# =============================================================================
TELEGRAM_BOT_TOKEN = "8740740753:AAFyJ_T_r4JHJDR_oJEwo01ui68IQHHZAtA" 
TELEGRAM_CHAT_ID = "5811007750"


# 1. 야후 파이낸스 데이터 수집
def fetch_yahoo_data(symbol, days=1095):
    now = datetime.now()
    start = now - timedelta(days=days)
    p1, p2 = int(time.mktime(start.timetuple())), int(time.mktime(now.timetuple()))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()["chart"]["result"][0]
        timestamps = data.get("timestamp", [])
        close_prices = data["indicators"]["quote"][0].get("close", [])

        dates = pd.to_datetime(timestamps, unit="s").normalize()
        s = pd.Series(close_prices, index=dates, name=symbol).dropna()
        return s[~s.index.duplicated(keep="last")]
    except Exception as e:
        print(f"[{symbol}] 야후 데이터 수집 실패: {e}")
        return None


# 2. 조지아 중앙은행(NBG) 주간 데이터 수집
def fetch_nbg_single_date(dt):
    date_str = dt.strftime("%Y-%m-%d")
    url = f"https://nbg.gov.ge/gw/api/ct/monetarypolicy/currencies/en/json/?date={date_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        res = requests.get(url, headers=headers, timeout=5, verify=False)
        res.raise_for_status()
        data = res.json()
        if data and len(data) > 0:
            for curr in data[0].get("currencies", []):
                if curr.get("code") == "USD":
                    return dt, float(curr["rate"])
    except Exception:
        pass
    return dt, None


def fetch_nbg_gel_weekly_data(weeks=156):
    print(f"[USDGEL] NBG 주간 데이터 병렬 수집 중 ({weeks}주)...")
    weekly_dates = pd.date_range(end=datetime.now(), periods=weeks, freq="W-FRI")

    records = {}
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = executor.map(fetch_nbg_single_date, weekly_dates)
        for dt, rate in results:
            if rate is not None:
                records[dt] = rate

    if not records:
        print("[USDGEL] NBG 데이터 수집 실패")
        return None

    s = pd.Series(records, name="GEL").sort_index()
    return s.ffill().bfill()


# 3. RSI 계산
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# 4. 머신러닝 예측 모듈
def analyze_currency_enhanced(df_weekly, currency_col, horizon=4):
    df = df_weekly.copy()

    df[f"{currency_col}_Ret_4W"] = df[currency_col].pct_change(horizon)
    df[f"{currency_col}_MA12_Ratio"] = (
        df[currency_col] / df[currency_col].rolling(12).mean() - 1
    )
    df[f"{currency_col}_MA24_Ratio"] = (
        df[currency_col] / df[currency_col].rolling(24).mean() - 1
    )
    df[f"{currency_col}_RSI14"] = calculate_rsi(df[currency_col], period=14)
    df[f"{currency_col}_Vol12W"] = (
        df[currency_col].pct_change().rolling(12).std()
    )

    df["DXY_Ret_4W"] = df["DXY"].pct_change(horizon)
    df["SPX_Ret_4W"] = df["SPX"].pct_change(horizon)
    df["TNX_Ret_4W"] = df["TNX"].pct_change(horizon)
    df["TNX_Level"] = df["TNX"]
    df["OIL_Ret_4W"] = df["OIL"].pct_change(horizon)
    df["VIX_Level"] = df["VIX"]
    df["VIX_Change_4W"] = df["VIX"].pct_change(horizon)

    df["Target_4W"] = (df[currency_col].shift(-horizon) > df[currency_col]).astype(int)

    feature_cols = [
        f"{currency_col}_Ret_4W",
        f"{currency_col}_MA12_Ratio",
        f"{currency_col}_MA24_Ratio",
        f"{currency_col}_RSI14",
        f"{currency_col}_Vol12W",
        "DXY_Ret_4W",
        "SPX_Ret_4W",
        "TNX_Ret_4W",
        "TNX_Level",
        "OIL_Ret_4W",
        "VIX_Level",
        "VIX_Change_4W",
    ]

    df_clean = df.dropna(subset=feature_cols)
    train_df = df_clean.iloc[:-horizon].dropna()

    X = train_df[feature_cols]
    y = train_df["Target_4W"]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        cv_scores.append(accuracy_score(y_te, preds))

    avg_acc = np.mean(cv_scores)

    final_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    final_model.fit(X, y)

    latest_X = df_clean[feature_cols].iloc[[-1]]
    latest_close = df_clean[currency_col].iloc[-1]

    probs = final_model.predict_proba(latest_X)[0]
    classes = list(final_model.classes_)
    up_prob = probs[classes.index(1)] * 100 if 1 in classes else 0.0

    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return latest_close, avg_acc, up_prob, importances, df_clean


# 5. 텔레그램 전송 함수
def send_telegram_report(bot_token, chat_id, report_text, image_path="dashboard.png"):
    if not bot_token or not chat_id or "여기에_" in bot_token or "여기에_" in chat_id:
        print("\n[텔레그램 전송 건너뜀] BOT_TOKEN 또는 CHAT_ID가 올바르게 설정되지 않았습니다.")
        return

    print("\n텔레그램 전송 중...")

    text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        res_text = requests.post(
            text_url, data={"chat_id": chat_id, "text": report_text}, timeout=10
        )
        if not res_text.ok:
            print(f"\n[텔레그램 서버 응답 오류 상세]: {res_text.text}")
            res_text.raise_for_status()

        photo_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(image_path, "rb") as photo:
            res_photo = requests.post(
                photo_url,
                data={"chat_id": chat_id},
                files={"photo": photo},
                timeout=20,
            )
            if not res_photo.ok:
                print(f"\n[텔레그램 이미지 응답 오류 상세]: {res_photo.text}")
                res_photo.raise_for_status()

        print("[성공] 텔레그램 메세지 및 대시보드 차트가 정상 전송되었습니다.")
    except Exception as e:
        print(f"[오류] 텔레그램 전송 실패: {e}")


# 6. 메인 실행 함수
def main():
    os_name = platform.system()
    plt.rc(
        "font",
        family=(
            "Malgun Gothic"
            if os_name == "Windows"
            else "AppleGothic"
            if os_name == "Darwin"
            else "NanumGothic"
        ),
    )
    plt.rcParams["axes.unicode_minus"] = False

    print("확장 매크로 및 글로벌 데이터 수집 중...")
    krw = fetch_yahoo_data("USDKRW=X", 1095)
    gel = fetch_nbg_gel_weekly_data(weeks=156)
    dxy = fetch_yahoo_data("DX-Y.NYB", 1095)
    spx = fetch_yahoo_data("^GSPC", 1095)
    tnx = fetch_yahoo_data("^TNX", 1095)
    oil = fetch_yahoo_data("CL=F", 1095)
    vix = fetch_yahoo_data("^VIX", 1095)

    if any(v is None or len(v) == 0 for v in [krw, gel, dxy, spx, tnx, oil, vix]):
        print("[오류] 필수 데이터 수집에 실패했습니다.")
        return

    krw_w = krw.resample("W-FRI").last()
    dxy_w = dxy.resample("W-FRI").last()
    spx_w = spx.resample("W-FRI").last()
    tnx_w = tnx.resample("W-FRI").last()
    oil_w = oil.resample("W-FRI").last()
    vix_w = vix.resample("W-FRI").last()

    df_weekly = (
        pd.DataFrame(
            {
                "KRW": krw_w,
                "GEL": gel,
                "DXY": dxy_w,
                "SPX": spx_w,
                "TNX": tnx_w,
                "OIL": oil_w,
                "VIX": vix_w,
            }
        )
        .ffill()
        .bfill()
        .dropna()
    )

    krw_close, krw_acc, krw_prob, krw_imp, df_krw = analyze_currency_enhanced(
        df_weekly, "KRW"
    )
    gel_close, gel_acc, gel_prob, gel_imp, df_gel = analyze_currency_enhanced(
        df_weekly, "GEL"
    )

    # 텍스트 리포트 생성
    report_text = f"""==================================================
 [고도화 매크로/기술적 지표 결합 1개월 환율 예측 리포트]
==================================================
1. 한국 원화 (USDKRW)
   - 현재 환율: {krw_close:.2f}원
   - 시계열 교차검증 평균 적중률: {krw_acc * 100:.1f}%
   - 향후 1개월 내 상승 확률: {krw_prob:.1f}%
   - 추세 시그널: {'[상승 추세 (Bullish)]' if krw_prob >= 50 else '[하락 추세 지속 (Bearish)]'}
--------------------------------------------------
2. 조지아 라리화 (USDGEL - NBG 실시간 공식 연동)
   - 현재 환율: {gel_close:.4f} GEL
   - 시계열 교차검증 평균 적중률: {gel_acc * 100:.1f}%
   - 향후 1개월 내 상승 확률: {gel_prob:.1f}%
   - 추세 시그널: {'[상승 추세 (Bullish)]' if gel_prob >= 50 else '[하락 추세 지속 (Bearish)]'}
=================================================="""

    print("\n" + report_text + "\n")

    # 8종 확장 대시보드 그래프 생성 (4행 2열)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(
        "다중 통화 1개월 추세 예측 및 매크로 종합 분석 대시보드",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    # 1) USDKRW 추이
    axes[0, 0].plot(df_weekly.index, df_weekly["KRW"], color="black", lw=2, label="USDKRW 실제 환율")
    axes[0, 0].plot(df_weekly.index, df_weekly["KRW"].rolling(12).mean(), color="orange", ls="--", label="MA12")
    axes[0, 0].set_title(f"USDKRW (적중률: {krw_acc * 100:.1f}%, 상승확률: {krw_prob:.1f}%)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left")

    # 2) USDGEL 추이
    axes[0, 1].plot(df_weekly.index, df_weekly["GEL"], color="navy", lw=2, label="USDGEL 실제 환율")
    axes[0, 1].plot(df_weekly.index, df_weekly["GEL"].rolling(12).mean(), color="red", ls="--", label="MA12")
    axes[0, 1].set_title(f"USDGEL (적중률: {gel_acc * 100:.1f}%, 상승확률: {gel_prob:.1f}%)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper left")

    # 3) KRW 중요도
    top_krw_imp = krw_imp.head(6)
    axes[1, 0].barh(top_krw_imp.index[::-1], top_krw_imp.values[::-1] * 100, color="teal")
    axes[1, 0].set_title("USDKRW 예측 기여 변수 Top 6 (%)")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # 4) GEL 중요도
    top_gel_imp = gel_imp.head(6)
    axes[1, 1].barh(top_gel_imp.index[::-1], top_gel_imp.values[::-1] * 100, color="darkslateblue")
    axes[1, 1].set_title("USDGEL 예측 기여 변수 Top 6 (%)")
    axes[1, 1].grid(True, alpha=0.3, axis="x")

    # 5) 글로벌 매크로 지표 추이 (Oil, TNX, VIX)
    ax_oil = axes[2, 0]
    ax_tnx = ax_oil.twinx()
    ax_vix = ax_oil.twinx()
    ax_vix.spines["right"].set_position(("outward", 55))

    l1 = ax_oil.plot(df_weekly.index, df_weekly["OIL"], color="green", alpha=0.8, label="WTI 유가($)")
    l2 = ax_tnx.plot(df_weekly.index, df_weekly["TNX"], color="blue", alpha=0.8, lw=1.8, label="미10년물 금리(%)")
    l3 = ax_vix.plot(df_weekly.index, df_weekly["VIX"], color="crimson", ls=":", alpha=0.8, label="VIX 공포지수")

    ax_oil.set_ylabel("WTI 유가 ($)", color="green")
    ax_tnx.set_ylabel("미10년물 금리 (%)", color="blue")
    ax_vix.set_ylabel("VIX 지수", color="crimson")

    ax_oil.tick_params(axis="y", labelcolor="green")
    ax_tnx.tick_params(axis="y", labelcolor="blue")
    ax_vix.tick_params(axis="y", labelcolor="crimson")

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_oil.legend(lines, labels, loc="upper left")
    ax_oil.set_title("글로벌 주요 매크로 지표 추이 (개별 스케일 적용)")
    ax_oil.grid(True, alpha=0.3)

    # 6) RSI
    axes[2, 1].plot(df_krw.index, df_krw["KRW_RSI14"], color="black", label="원화 RSI(14)")
    axes[2, 1].plot(df_gel.index, df_gel["GEL_RSI14"], color="navy", label="라리화 RSI(14)")
    axes[2, 1].axhline(70, color="red", ls="--", alpha=0.5)
    axes[2, 1].axhline(30, color="blue", ls="--", alpha=0.5)
    axes[2, 1].set_title("RSI 과매수(70 이상) / 과매도(30 이하) 보조지표")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(loc="upper left")

    # 7) [신규 추가] DXY & S&P 500 추이
    ax_dxy = axes[3, 0]
    ax_spx = ax_dxy.twinx()
    
    l_dxy = ax_dxy.plot(df_weekly.index, df_weekly["DXY"], color="purple", lw=1.8, label="달러 인덱스 (DXY)")
    l_spx = ax_spx.plot(df_weekly.index, df_weekly["SPX"], color="darkgreen", ls="--", lw=1.5, label="S&P 500 지수")
    
    ax_dxy.set_ylabel("DXY Index", color="purple")
    ax_spx.set_ylabel("S&P 500 Index", color="darkgreen")
    ax_dxy.tick_params(axis="y", labelcolor="purple")
    ax_spx.tick_params(axis="y", labelcolor="darkgreen")
    
    lines_dxy = l_dxy + l_spx
    labels_dxy = [l.get_label() for l in lines_dxy]
    ax_dxy.legend(lines_dxy, labels_dxy, loc="upper left")
    ax_dxy.set_title("글로벌 달러 강세(DXY) 및 미국 증시(S&P 500) 추이")
    ax_dxy.grid(True, alpha=0.3)

    # 8) [신규 추가] 12주 이동 연율화 변동성 비교
    krw_vol = df_weekly["KRW"].pct_change().rolling(12).std() * np.sqrt(52) * 100
    gel_vol = df_weekly["GEL"].pct_change().rolling(12).std() * np.sqrt(52) * 100
    
    axes[3, 1].plot(df_weekly.index, krw_vol, color="black", lw=1.5, label="원화 변동성 (%)")
    axes[3, 1].plot(df_weekly.index, gel_vol, color="navy", lw=1.5, label="라리화 변동성 (%)")
    axes[3, 1].set_title("12주 이동 연율화 변동성 (%) 비교")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    image_filename = "dashboard.png"
    plt.savefig(image_filename, dpi=300, bbox_inches="tight")

    send_telegram_report(
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, report_text, image_filename
    )

    plt.show()


if __name__ == "__main__":
    main()
