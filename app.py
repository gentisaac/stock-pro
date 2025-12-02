import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# --- 頁面設定 ---
st.set_page_config(page_title="第二層思維戰情室 Ultimate", layout="wide", page_icon="🦅")

st.title("🦅 第二層思維搶跑戰情室 V8.0")
st.markdown("""
**核心策略：** 尋找市場恐慌、乖離過大、但主力在關鍵支撐位（L2）有防守跡象的標的。
* **L1 (大眾):** 均線安全區
* **L2 (搶跑):** **(實戰進場點)** 現價低於此價位即可分批佈局。
* **L3 (接血):** **(極端警報)** 主力獵殺止損區，若跌破此處代表恐慌極致。
""")

# --- 獲取指數成分股函數 (含強力備援) ---
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(r.text)
        for df in dfs:
            if 'Symbol' in df.columns: return df['Symbol'].tolist()
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "LLY", "V", "JPM", "XOM", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "MRK", "COST", "ABBV", "CVX", "CRM", "BAC", "KO"] # Fallback
    except:
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"] # Fallback

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
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ASML", "COST", "PEP", "CSCO", "NFLX", "AMD", "ADBE", "TMUS", "INTC", "QCOM", "TXN", "AMGN", "HON", "INTU", "BKNG"] # Fallback
    except:
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"]

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 掃描設定")

# 自選清單
st.sidebar.subheader("👑 我的自選 (必看)")
default_custom = "NVDA, TSLA, MSTR, SMR, PLTR"
user_custom_str = st.sidebar.text_area("輸入代號", default_custom, height=70)
custom_tickers = [x.strip().upper() for x in user_custom_str.split(',') if x.strip()]

st.sidebar.divider()

# 掃描模式
st.sidebar.subheader("🔍 全市場掃描模式")
scan_mode = st.sidebar.radio(
    "選擇掃描範圍:",
    ("手動輸入清單", "S&P 500 成分股 (約3分鐘)", "Nasdaq 100 成分股 (約1分鐘)")
)

pool_tickers = []
if scan_mode == "手動輸入清單":
    default_pool = "AAPL, AMD, META, AMZN, MSFT, GOOGL, NFLX, COIN, MARA, PLTR, SOFI, UBER, DIS, PYPL, SQ, SHOP, GME, HOOD, AFRM, UPST, RIOT, CLSK"
    user_pool_str = st.sidebar.text_area("輸入掃描清單", default_pool, height=150)
    pool_tickers = [x.strip().upper() for x in user_pool_str.split(',') if x.strip()]
elif scan_mode == "S&P 500 成分股 (約3分鐘)":
    with st.sidebar.status("下載 S&P 500 名單中..."):
        pool_tickers = get_sp500_tickers()
        st.write(f"取得 {len(pool_tickers)} 檔")
elif scan_mode == "Nasdaq 100 成分股 (約1分鐘)":
    with st.sidebar.status("下載 Nasdaq 100 名單中..."):
        pool_tickers = get_nasdaq100_tickers()
        st.write(f"取得 {len(pool_tickers)} 檔")

run_btn = st.sidebar.button("🚀 開始掃描", type="primary")

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
        df = yf.download(t, period="50d", interval="1d", progress=False)
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
        recent_low = df['Low'].tail(10).min()
        l2_entry = max(sma20 - 2*std20, recent_low * 1.005)
        l3_entry = recent_low * 0.975
        
        # 狀態標記
        status = ""
        if curr['Close'] <= l3_entry: status = "🚨接血"
        elif curr['Close'] <= l2_entry: status = "🟢可進場"
        
        return {
            "代號": t, 
            "狀態": status, # 新增狀態欄位
            "現價": round(curr['Close'], 2), 
            "總分": total_score,
            "RSI": round(curr['RSI'], 1), "RSI分": s_rsi,
            "KD": round(curr['K'], 1), "KD分": s_kd,
            "量能倍數": round(vol_ratio, 1), "量能分": s_vol,
            "L2搶跑價": round(l2_entry, 2),
            "止損價": round(recent_low * 0.985, 2),
            "L3接血價": round(l3_entry, 2),
            "Data": df.tail(40)
        }
    except: return None

def render_stock_card(row, alert_type="normal"):
    t = row['代號']
    status = row['狀態']
    
    # 標題區塊
    if alert_type == "L3":
        st.error(f"🚨 **{t}** 崩盤警報！現價 {row['現價']} 已跌破 L3 接血價！")
    elif alert_type == "L2":
        st.success(f"🟢 **{t}** 買點浮現！現價 {row['現價']} 已低於 L2 搶跑價。")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if alert_type == "normal": st.markdown(f"### {t}")
        
        # 分數顯示
        score = row['總分']
        st.metric("綜合評分", f"{score} / 20", delta=status if status else None)
        
        st.write("---")
        st.markdown(f"**🟢 L2 進場:** `{row['L2搶跑價']}`")
        st.markdown(f"**🔴 嚴格止損:** `{row['止損價']}`")
        st.markdown(f"**🟣 L3 接血:** `{row['L3接血價']}`")
        st.write("---")
        st.caption(f"RSI: {row['RSI']} ({row['RSI分']}分)")
        st.caption(f"KD: {row['KD']} ({row['KD分']}分)")
        st.caption(f"量能: {row['量能倍數']}倍 ({row['量能分']}分)")

    with col1:
        df = row['Data']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=t), row=1, col=1)
        
        fig.add_hline(y=row['L2搶跑價'], line_width=2, line_dash="dash", line_color="#00FF00", row=1, col=1)
        fig.add_hline(y=row['止損價'], line_width=2, line_color="#FF0000", row=1, col=1)
        fig.add_hline(y=row['L3接血價'], line_width=2, line_dash="dot", line_color="purple", row=1, col=1)
        
        colors = ['red' if r['Open'] > r['Close'] else 'green' for k, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        
        y_min = min(df['Low'].min(), row['L3接血價']) * 0.98
        y_max = df['High'].max() * 1.02
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_rangeslider_visible=False, yaxis=dict(range=[y_min, y_max]))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

# --- 主程式邏輯 ---
if run_btn:
    # 1. 自選清單
    if custom_tickers:
        st.header(f"👑 我的自選關注 ({len(custom_tickers)})")
        with st.spinner("分析自選股中..."):
            for t in custom_tickers:
                res = analyze_stock(t)
                if res: render_stock_card(res, alert_type="L2" if res['狀態'] else "normal")

    # 2. 市場掃描
    if pool_tickers:
        st.header(f"🏆 {scan_mode} 掃描結果")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 分類清單
        l3_list = []      # 跌破 L3
        l2_list = []      # 跌破 L2
        high_score_list = [] # 單純高分
        
        total = len(pool_tickers)
        
        for i, t in enumerate(pool_tickers):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"掃描中 ({i+1}/{total}): {t} ...")
            
            if t in custom_tickers: continue 
            
            res = analyze_stock(t)
            if res:
                # 邏輯分流
                if res['現價'] <= res['L3接血價']:
                    l3_list.append(res)
                elif res['現價'] <= res['L2搶跑價']:
                    l2_list.append(res)
                
                # 只要分數夠高，也放入觀察名單 (但不重複放 L3/L2 的)
                if res['總分'] >= 10:
                    high_score_list.append(res)
        
        progress_bar.empty()
        status_text.empty()

        # === 區塊 1: L3 極限警報 ===
        if l3_list:
            st.markdown("### 🚨 L3 崩盤極限警報 (帶血籌碼)")
            for row in l3_list:
                render_stock_card(row, alert_type="L3")

        # === 區塊 2: L2 直接進場 Top 10 (依照總分排序) ===
        if l2_list:
            st.markdown("### 🟢 L2 直接進場 Top 10 (現價已低於買點)")
            st.info("以下標的雖然未跌到 L3，但已進入 L2 搶跑區，且依照綜合評分排序，是勝率較高的選擇。")
            
            df_l2 = pd.DataFrame(l2_list)
            df_l2 = df_l2.sort_values(by="總分", ascending=False).head(10)
            
            # 表格顯示 (隱藏 index, 顯示狀態)
            st.dataframe(
                df_l2.drop(columns=["Data"]).style.background_gradient(subset=['總分'], cmap='RdYlGn').hide(axis="index"), 
                use_container_width=True
            )
            
            for index, row in df_l2.iterrows():
                # 避免跟 L3 重複顯示
                if row['代號'] not in [x['代號'] for x in l3_list]:
                    render_stock_card(row, alert_type="L2")

        # === 區塊 3: 高分潛力 Top 10 (不一定能買，但值得看) ===
        if high_score_list:
            st.markdown("### 📊 高分潛力觀察名單 Top 10 (總分 >= 10)")
            st.caption("此區為恐慌分數高的標的，但價格未必已跌破 L2，請耐心等待價格落入區間。")
            
            df_pool = pd.DataFrame(high_score_list)
            df_pool = df_pool.sort_values(by="總分", ascending=False).head(10)
            
            st.dataframe(
                df_pool.drop(columns=["Data"]).style.background_gradient(subset=['總分'], cmap='RdYlGn').hide(axis="index"), 
                use_container_width=True
            )
            # 這裡不顯示圖卡，以免頁面太長，圖卡只留給 L3 和 L2 這些能動作的
        
        if not l3_list and not l2_list and not high_score_list:
            st.warning("掃描完成。目前市場情緒平穩，沒有觸發任何策略信號。")
            
    elif not pool_tickers and scan_mode != "手動輸入清單":
        st.error("無法下載成分股名單，將使用內建備用清單進行掃描。")
        
else:
    st.info("👈 請在左側選擇掃描範圍，並點擊「🚀 開始掃描」按鈕。")
