import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# --- 頁面設定 ---
st.set_page_config(page_title="第二層思維戰情室 V9.2", layout="wide", page_icon="🦅")

st.title("🦅 第二層思維戰情室 V9.2")
st.markdown("""
**策略邏輯：**
* **買入 (L2):** 尋找恐慌乖離，在支撐位搶反彈。
* **賣出 (High):** 當股價回歸均值 (L1) 時減倉，突破上緣時全出。
""")

# --- 獲取指數成分股函數 ---
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(r.text)
        for df in dfs:
            if 'Symbol' in df.columns: return df['Symbol'].tolist()
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"] 
    except: return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"]

@st.cache_data(ttl=3600)
def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(r.text)
        for df in dfs:
            if 'Ticker' in df.columns: return df['Ticker'].tolist()
            elif 'Symbol' in df.columns: return df['Symbol'].tolist()
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"]
    except: return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"]

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 設定面板")

# 1. 持股診斷區
st.sidebar.subheader("💼 我的持股/自選 (診斷賣點)")
default_custom = "NVDA, TSLA, PLTR, SOFI"
user_custom_str = st.sidebar.text_area("輸入代號", default_custom, height=70)
custom_tickers = [x.strip().upper() for x in user_custom_str.split(',') if x.strip()]

st.sidebar.divider()

# 2. 市場掃描區
st.sidebar.subheader("🔍 市場掃描 (找買點)")
scan_mode = st.sidebar.radio(
    "選擇範圍:",
    ("不掃描 (只看持股)", "手動輸入清單", "S&P 500 (約3分鐘)", "Nasdaq 100 (約1分鐘)")
)

pool_tickers = []
if scan_mode == "手動輸入清單":
    default_pool = "AAPL, AMD, META, AMZN, MSFT, GOOGL, NFLX, COIN, MARA, PLTR, SOFI, UBER, DIS, PYPL, SQ, SHOP"
    user_pool_str = st.sidebar.text_area("輸入掃描清單", default_pool, height=150)
    pool_tickers = [x.strip().upper() for x in user_pool_str.split(',') if x.strip()]
elif scan_mode == "S&P 500 (約3分鐘)":
    with st.sidebar.status("下載 S&P 500 名單..."):
        pool_tickers = get_sp500_tickers()
        st.write(f"取得 {len(pool_tickers)} 檔")
elif scan_mode == "Nasdaq 100 (約1分鐘)":
    with st.sidebar.status("下載 Nasdaq 100 名單..."):
        pool_tickers = get_nasdaq100_tickers()
        st.write(f"取得 {len(pool_tickers)} 檔")

run_btn = st.sidebar.button("🚀 開始分析", type="primary")

# --- 核心計算函數 ---
def calculate_indicators(df):
    df = df.copy()
    cols = ['Close', 'High', 'Low', 'Volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()

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
        
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        
        recent_low = df['Low'].tail(10).min()
        
        l1_sell_target = sma20 * 0.99
        extreme_sell_target = upper_band * 0.98
        
        l2_entry = max(lower_band, recent_low * 1.005)
        l3_entry = recent_low * 0.975
        stop_loss = recent_low * 0.985
        
        price = curr['Close']
        
        # 計算低於 L2 的幅度 (百分比)
        l2_discount = 0.0
        if price < l2_entry:
            l2_discount = (l2_entry - price) / l2_entry * 100
        
        signal = "觀望"
        signal_color = "gray"
        advice = "等待機會"
        
        if price < stop_loss:
            signal = "🛑 止損離場"
            signal_color = "red"
            advice = "跌破防守點"
        elif price >= extreme_sell_target or curr['RSI'] > 68:
            signal = "🔴 獲利全出"
            signal_color = "red"
            advice = "接近極限壓力/過熱"
        elif price >= l1_sell_target:
            signal = "🟠 獲利減倉"
            signal_color = "orange"
            advice = "接近均線壓力"
        elif price <= l3_entry:
            signal = "🚨 L3 接血"
            signal_color = "violet"
            advice = "極端恐慌區"
        elif price <= l2_entry:
            signal = "🟢 L2 進場"
            signal_color = "green"
            advice = "搶跑支撐區"
            
        return {
            "代號": t, 
            "訊號": signal, "顏色": signal_color, "建議": advice,
            "現價": round(curr['Close'], 2), 
            "總分": total_score,
            "RSI": round(curr['RSI'], 1), 
            "KD": round(curr['K'], 1), 
            "量能": round(vol_ratio, 1),
            "L1賣點(均線)": round(l1_sell_target, 2),
            "L2搶跑價": round(l2_entry, 2),
            "L3接血價": round(l3_entry, 2),
            "止損價": round(stop_loss, 2),
            "極限賣點(上緣)": round(extreme_sell_target, 2),
            "L2乖離": l2_discount, # 新增這個原始數值方便排序
            "Data": df.tail(45)
        }
    except: return None

def render_stock_card(row, mode="scan"):
    t = row['代號']
    signal = row['訊號']
    
    if "全出" in signal or "止損" in signal:
        st.error(f"[{t}] {signal}：{row['建議']}")
    elif "減倉" in signal:
        st.warning(f"[{t}] {signal}：{row['建議']}")
    elif "進場" in signal or "接血" in signal:
        st.success(f"[{t}] {signal}：{row['建議']}")
    elif mode == "portfolio":
        st.info(f"[{t}] {signal}：{row['建議']}")

    col1, col2 = st.columns([3, 1])
    
    with col2:
        if mode == "scan" and "觀望" in signal: st.markdown(f"### {t}")
        
        st.metric("訊號判定", signal)
        st.write("---")
        
        if mode == "portfolio":
            st.markdown(f"**🎯 賣點:** `{row['L1賣點(均線)']}`")
            st.markdown(f"**🚀 極限:** `{row['極限賣點(上緣)']}`")
            st.markdown(f"**🛡️ 止損:** `{row['止損價']}`")
        else:
            # 如果低於 L2，顯示便宜了多少
            discount_str = ""
            if row['L2乖離'] > 0:
                discount_str = f"(📉 便宜 {row['L2乖離']:.2f}%)"
            
            st.markdown(f"**🟢 L2:** `{row['L2搶跑價']}` {discount_str}")
            st.markdown(f"**🟣 L3:** `{row['L3接血價']}`")
            st.markdown(f"**🛡️ 止損:** `{row['止損價']}`")
            
        st.write("---")
        st.caption(f"RSI: {row['RSI']} | KD: {row['KD']} | 量: {row['量能']}倍")

    with col1:
        df = row['Data']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=t), row=1, col=1)
        
        if mode == "portfolio":
            fig.add_hline(y=row['L1賣點(均線)'], line_width=2, line_dash="dash", line_color="orange", row=1, col=1)
            fig.add_hline(y=row['極限賣點(上緣)'], line_width=2, line_color="red", row=1, col=1)
            fig.add_hline(y=row['止損價'], line_width=2, line_color="gray", row=1, col=1)
        else:
            fig.add_hline(y=row['L2搶跑價'], line_width=2, line_dash="dash", line_color="#00FF00", row=1, col=1)
            fig.add_hline(y=row['L3接血價'], line_width=2, line_dash="dot", line_color="purple", row=1, col=1)
            fig.add_hline(y=row['止損價'], line_width=2, line_color="#FF0000", row=1, col=1)
            
        colors = ['red' if r['Open'] > r['Close'] else 'green' for k, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

# --- 主程式邏輯 ---
if run_btn:
    
    # 1. 持股診斷
    if custom_tickers:
        st.header(f"💼 我的持股診斷 ({len(custom_tickers)})")
        with st.spinner("診斷中..."):
            for t in custom_tickers:
                res = analyze_stock(t)
                if res: render_stock_card(res, mode="portfolio")

    # 2. 市場掃描
    if scan_mode != "不掃描 (只看持股)" and pool_tickers:
        st.header(f"🏆 {scan_mode} 潛力買點掃描")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        buy_list = []
        watch_list = []
        
        total = len(pool_tickers)
        for i, t in enumerate(pool_tickers):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"掃描中 ({i+1}/{total}): {t} ...")
            
            if t in custom_tickers: continue 
            
            res = analyze_stock(t)
            if res:
                if "進場" in res['訊號'] or "接血" in res['訊號']:
                    buy_list.append(res)
                elif res['總分'] >= 10:
                    watch_list.append(res)
        
        progress_bar.empty()
        status_text.empty()

        # A. 直接進場區 (新增 低於L2幅度 欄位)
        if buy_list:
            st.markdown("### 🟢 可進場標的 (現價低於 L2)")
            df_buy = pd.DataFrame(buy_list).sort_values(by="總分", ascending=False)
            
            # 格式化顯示：把小數點變成百分比字串，方便閱讀
            df_buy['低於L2幅度'] = df_buy['L2乖離'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(
                df_buy[['代號', '現價', '總分', '低於L2幅度', 'L2搶跑價', 'L3接血價', 'RSI', 'KD', '量能']], 
                use_container_width=True,
                hide_index=True 
            )
            
            for index, row in df_buy.iterrows():
                render_stock_card(row, mode="scan")
        
        # B. 高分觀察區
        if watch_list:
            st.markdown("### 📊 高分潛力 Top 10 (總分 >= 10)")
            df_watch = pd.DataFrame(watch_list).sort_values(by="總分", ascending=False).head(10)
            st.dataframe(
                df_watch[['代號', '現價', '總分', 'L2搶跑價', 'RSI', 'KD', '量能']], 
                use_container_width=True,
                hide_index=True
            )
        
        if not buy_list and not watch_list:
            st.warning("掃描完成，無符合條件的標的。")

else:
    st.info("👈 請在左側輸入持股，並點擊「🚀 開始分析」按鈕。")
