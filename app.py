import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# --- 頁面設定 ---
st.set_page_config(page_title="第二層思維戰情室 Ultimate", layout="wide", page_icon="🦅")

st.title("🦅 第二層思維搶跑戰情室 Ultimate")
st.markdown("""
**核心策略：** 尋找市場恐慌、乖離過大、但主力在關鍵支撐位（L2）有防守跡象的標的。
* **L1 (大眾):** 均線安全區
* **L2 (搶跑):** 我們的主戰場 (極窄止損)
* **L3 (接血):** **(極端警報)** 防範主力獵殺止損的更深點位，若現價低於此，代表機會與風險並存。
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
        return []
    except: return []

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
        return []
    except: return []

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 掃描設定")

# 自選清單
st.sidebar.subheader("👑 我的自選 (必看)")
default_custom = "NVDA, TSLA, MSTR, SMR"
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
        if not pool_tickers: pool_tickers = ["AAPL", "MSFT", "NVDA"] # Fallback
        st.write(f"取得 {len(pool_tickers)} 檔")
elif scan_mode == "Nasdaq 100 成分股 (約1分鐘)":
    with st.sidebar.status("下載 Nasdaq 100 名單中..."):
        pool_tickers = get_nasdaq100_tickers()
        if not pool_tickers: pool_tickers = ["AAPL", "MSFT", "NVDA"] # Fallback
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
        
        # 定義 L3 接血價 (前低之下 2.5%)
        l3_entry = recent_low * 0.975
        
        return {
            "代號": t, "現價": round(curr['Close'], 2), "總分": total_score,
            "RSI": round(curr['RSI'], 1), "RSI分": s_rsi,
            "KD": round(curr['K'], 1), "KD分": s_kd,
            "量能倍數": round(vol_ratio, 1), "量能分": s_vol,
            "L2搶跑價": round(l2_entry, 2),
            "止損價": round(recent_low * 0.985,
