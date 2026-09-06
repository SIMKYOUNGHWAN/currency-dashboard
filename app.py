# 4) [신규 추가] 기술적 보조지표 (RSI & 변동성) - 폰트 깨짐 및 X축 겹침 수정
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
    
    # X축 날짜 간격 자동 정돈 및 회전 적용
    fig_vol.autofmt_xdate(rotation=30)
    fig_vol.tight_layout()
    st.pyplot(fig_vol)
