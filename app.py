import streamlit as st
import pandas as pd
import numpy as np
import time

# 設定網頁標題與排版
st.set_page_config(page_title="RWA 動態估值與風險評級系統", layout="wide")

# 標題與學術提交資訊
st.title("📊 AI 驅動 RWA 動態估值與風險評級系統 (POC)")
st.markdown("**開發者：喬睿宇 (學號：113ab0044) | 國立臺北科技大學 資訊與財金管理系**")
st.markdown("本系統模擬將實質資產 (RWA) 如不動產、債權等數據，透過機器學習模型進行動態估值，並給予對應的風險評級。")
st.divider()

# 模擬 RWA 資產資料庫
@st.cache_data
def load_mock_rwa_data():
    data = {
        "資產代碼": ["RWA-RE-001", "RWA-RE-002", "RWA-BD-001", "RWA-BD-002"],
        "資產類型": ["商辦大樓 (台北)", "住宅 (新北)", "企業應收帳款", "基礎建設債券"],
        "初始估值 (USD)": [15000000, 3200000, 500000, 8000000],
        "歷史波動率": [0.05, 0.08, 0.12, 0.03],
    }
    return pd.DataFrame(data)

df = load_mock_rwa_data()

# --- 亮點 1: 即時資產估值更新 ---
st.header("1. 📈 即時資產估值更新")
st.write("透過整合市場數據（如房價指數、利率），AI 模擬預測最新的資產價值。")

# 模擬 AI 動態估值計算 (加入隨機波動模擬動態市場)
np.random.seed(int(time.time()))
df['市場影響因子'] = np.random.uniform(-0.05, 0.08, len(df))
df['即時估值 (USD)'] = df['初始估值 (USD)'] * (1 + df['市場影響因子'])
df['即時估值 (USD)'] = df['即時估值 (USD)'].astype(int)

st.dataframe(df[['資產代碼', '資產類型', '初始估值 (USD)', '即時估值 (USD)']], use_container_width=True)

# --- 亮點 2: 信用評等（AAA–C）與風險分析 ---
st.header("2. 🛡️ 信用評等與風險分析 (AAA-C)")
st.write("根據資產的歷史波動率與即時市場影響因子，系統自動核算風險評級。")

def calculate_risk_rating(volatility, market_impact):
    # 簡單的決策樹邏輯模擬風險評級
    score = volatility * 100 - market_impact * 50
    if score < 2:
        return "AAA"
    elif score < 5:
        return "AA"
    elif score < 8:
        return "A"
    elif score < 12:
        return "BBB"
    else:
        return "C (高風險)"

df['信用評等'] = df.apply(lambda row: calculate_risk_rating(row['歷史波動率'], row['市場影響因子']), axis=1)

# 視覺化風險分佈
col1, col2 = st.columns([2, 1])
with col1:
    st.table(df[['資產代碼', '資產類型', '信用評等', '歷史波動率']])
with col2:
    rating_counts = df['信用評等'].value_counts()
    st.bar_chart(rating_counts)

# --- 亮點 3: 投資決策輔助系統 ---
st.header("3. 💡 投資決策輔助系統")
st.write("輸入你的風險偏好，AI 將為你過濾適合的 RWA 投資標的。")

risk_tolerance = st.selectbox("請選擇您的風險承受度：", ["保守型 (偏好 AAA, AA)", "穩健型 (偏好 A, BBB)", "積極型 (可接受 C)"])

st.subheader("推薦投資標的：")
if risk_tolerance == "保守型 (偏好 AAA, AA)":
    recommendation = df[df['信用評等'].isin(['AAA', 'AA'])]
elif risk_tolerance == "穩健型 (偏好 A, BBB)":
    recommendation = df[df['信用評等'].isin(['A', 'BBB'])]
else:
    recommendation = df[df['信用評等'] == 'C (高風險)']

if not recommendation.empty:
    st.dataframe(recommendation[['資產代碼', '資產類型', '即時估值 (USD)', '信用評等']])
else:
    st.info("目前市場上沒有完全符合您風險偏好的單一標的，建議調整配置比例。")
