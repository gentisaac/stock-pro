import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面設定 ---
st.set_page_config(page_title="第二層思維戰情室 Pro", layout="wide", page_icon="🦅")

st.title("🦅 第二層思維搶跑戰情室 Pro")
st.markdown("""
**核心策略：** 尋找市場恐慌、乖離過大、但主力在關鍵支撐位（L2）有防守跡象的標的。
* **L1 (大眾):** 均線安全區
* **L2 (搶跑):** 我們的主戰場 (極窄止損)
* **L3 (接血):** 防範主力獵殺止損的更深點位
""")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 戰情室設定")

# 1. 自選清單輸入區
st.sidebar.subheader("👑 我的自選關注 (必看)")
default_custom = "NVDA, TSLA, MSTR"
user_custom_str = st.sidebar.text_area("輸入代號 (無論分數高低都會顯示)", default_custom, height=80)
custom_tickers = [x.strip().upper() for x in user_custom_str.split(',') if x.strip()]

# 2. 系統掃描池輸入區
st.sidebar.subheader("🔍 市場掃描池 (選Top 10)")
default_pool = "AAPL, AMD, META, AMZN, MSFT, GOOGL, NFLX, COIN, MARA, PLTR, SOFI, UBER, DIS, PYPL, SQ, SHOP, GME, HOOD, AFRM, UPST, RIOT, CLSK"
user_pool_str = st.sidebar.text_area("輸入掃描範圍 (只顯示高分前10名)", default_pool, height=150)
pool_tickers = [x.strip().upper() for x in user_pool_str.split(',') if x.strip()]

run_btn = st.sidebar.button("🚀 更新戰情數據", type="primary")

# --- 核心計算函數 ---
def calculate_indicators(df):
    df = df.copy()
    cols = ['Close', 'High', 'Low', 'Volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

def get_score(value, type_, hist_current=0, hist_min=0):
    if pd.isna(value): return 1
    score = 1
    if type_ == 'RSI':
        if value < 20: score = 5
        elif value < 30: score = 4
        elif value < 40: score = 3
        elif value < 50: score = 2
    elif type_ == 'KD':
        if value < 15: score = 5
        elif value < 25: score = 4
        elif value < 35: score = 3
        elif value < 50: score = 2
    elif type_ == 'VOL':
        if value > 2.0: score = 5
        elif value > 1.5: score = 4
        elif value > 1.2: score = 3
        elif value > 1.0: score = 2
    elif type_ == 'MACD':
        if hist_current < 0:
            score = 3
            if hist_current < hist_min * 0.8: score = 5
            elif hist_current > hist_min and hist_current < 0: score = 4
    return score

def analyze_stock(t):
    try:
        df = yf.download(t, period="60d", interval="1d", progress=False)
        if df.empty or len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        # 簡單檢查數據完整性
        if pd.isna(curr.get('K')) or pd.isna(curr.get('RSI')): return None

        vol_ratio = curr['Volume'] / df['Volume'].mean()
        hist_min = df['Hist'].min()
        
        s_rsi = get_score(curr['RSI'], 'RSI')
        s_kd = get_score(curr['K'], 'KD')
        s_vol = get_score(vol_ratio, 'VOL')
        s_macd = get_score(0, 'MACD', curr['Hist'], hist_min)
        total_score = s_rsi + s_kd + s_vol + s_macd
        
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        std20 = df['Close'].rolling(20).std().iloc[-1]
        recent_low = df['Low'].tail(10).min()
        l2_entry = max(sma20 - 2*std20, recent_low * 1.005)
        
        return {
            "代號": t, "現價": round(curr['Close'], 2), "總分": total_score,
            "RSI": round(curr['RSI'], 1), "RSI分": s_rsi,
            "KD": round(curr['K'], 1), "KD分": s_kd,
            "量能倍數": round(vol_ratio, 1), "量能分": s_vol,
            "L2搶跑價": round(l2_entry, 2),
            "止損價": round(recent_low * 0.985, 2),
            "L3接血價": round(recent_low * 0.975, 2),
            "Data": df.tail(45)
        }
    except:
        return None

def render_stock_card(row, is_top10=False):
    t = row['代號']
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown(f"### {t}")
        # 分數顏色標示
        score = row['總分']
        color = "normal"
        if score >= 12: color = "off" # Streamlit metric color limitation workaround
        
        st.metric("綜合評分", f"{score} / 20", delta="🔥 強烈訊號" if score>=14 else None)
        
        st.write("---")
        st.markdown(f"**🟢 L2 進場:** `{row['L2搶跑價']}`")
        st.markdown(f"**🔴 嚴格止損:** `{row['止損價']}`")
        st.markdown(f"**🟣 L3 接血:** `{row['L3接血價']}`")
        st.write("---")
        
        # 指標詳細
        st.caption(f"RSI: {row['RSI']} ({row['RSI分']}分)")
        st.caption(f"KD: {row['KD']} ({row['KD分']}分)")
        st.caption(f"量能: {row['量能倍數']}倍 ({row['量能分']}分)")

    with col1:
        df = row['Data']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=t), row=1, col=1)
        
        # 畫線
        fig.add_hline(y=row['L2搶跑價'], line_width=2, line_dash="dash", line_color="#00FF00", row=1, col=1)
        fig.add_hline(y=row['止損價'], line_width=2, line_color="#FF0000", row=1, col=1)
        fig.add_hline(y=row['L3接血價'], line_width=2, line_dash="dot", line_color="purple", row=1, col=1)
        
        colors = ['red' if r['Open'] > r['Close'] else 'green' for k, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        
        y_min = min(df['Low'].min(), row['L3接血價']) * 0.98
        y_max = df['High'].max() * 1.02
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_rangeslider_visible=False, yaxis=dict(range=[y_min, y_max]))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

# --- 主程式邏輯 ---
if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. 處理自選清單
    custom_results = []
    if custom_tickers:
        status_text.text("正在分析自選清單...")
        for i, t in enumerate(custom_tickers):
            res = analyze_stock(t)
            if res: custom_results.append(res)
    
    # 2. 處理市場掃描
    pool_results = []
    if pool_tickers:
        for i, t in enumerate(pool_tickers):
            # 進度條計算
            progress = (i + 1) / len(pool_tickers)
            progress_bar.progress(progress)
            status_text.text(f"正在掃描市場: {t} ...")
            
            # 避免重複：如果已經在自選清單裡有了，掃描池就跳過，節省資源
            if t in custom_tickers: continue 
            
            res = analyze_stock(t)
            if res: pool_results.append(res)

    progress_bar.empty()
    status_text.empty()

    # --- 顯示結果 ---
    
    # A. 自選區塊
    if custom_results:
        st.header(f"👑 我的自選關注 ({len(custom_results)})")
        st.info("這裡是您指定的觀察標的，無論分數高低皆顯示。")
        for row in custom_results:
            render_stock_card(row)
    elif custom_tickers:
        st.warning("⚠️ 自選清單中的股票數據抓取失敗，請檢查代號是否正確。")

    # B. 系統推薦區塊
    if pool_results:
        st.header("🏆 系統掃描高分 Top 10")
        st.info("系統根據「恐慌度」與「超賣指標」篩選出的最佳搶跑機會。")
        
        # 排序並取前 10
        df_pool = pd.DataFrame(pool_results)
        df_pool = df_pool.sort_values(by="總分", ascending=False).head(10)
        
        # 先顯示一個總表
        st.dataframe(df_pool.drop(columns=["Data"]).style.background_gradient(subset=['總分'], cmap='RdYlGn').hide(axis="index"), use_container_width=True)
        st.write("") # 空行

        # 顯示個別卡片
        for index, row in df_pool.iterrows():
            render_stock_card(row, is_top10=True)
            
    elif pool_tickers:
        st.warning("市場掃描池沒有返回數據。")
        
else:
    st.info("👈 請在左側輸入您關注的股票，並點擊「🚀 更新戰情數據」按鈕。")
