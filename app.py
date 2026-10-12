import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. 頁面進階設定：使用寬螢幕佈局，設定深色/淺色主題支援
st.set_page_config(page_title="RWA 動態估值與風險評級", page_icon="📊", layout="wide")

# 2. 側邊欄設計 (Sidebar)：提升網頁層次感
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3256/3256083.png", width=100)
    st.title("控制面板")
    st.markdown("設定市場環境參數與風險容忍度")
    st.divider()

    # 互動式參數微調
    market_volatility = st.slider("市場整體波動率指數", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    risk_tolerance = st.selectbox("您的投資風險承受度：", ["保守型 (AAA, AA)", "穩健型 (A, BBB)", "積極型 (C 級高報酬)"])
    st.info("系統採用模擬 Random Forest 與 MLP 類神經網路架構進行 RWA 信用違約風險評估。")

# 3. 主畫面標題與學生資訊
st.title("📊 AI 驅動 RWA 動態估值與風險評級系統")
st.markdown("**開發者：喬睿宇 (學號：113ab0044) | 國立臺北科技大學 資訊與財金管理系**")
st.divider()

# 4. 資料生成與 ML 模擬邏輯
@st.cache_data
def load_and_predict_rwa_data(volatility):
    data = {
        "資產代碼": ["RWA-RE-001", "RWA-RE-002", "RWA-BD-001", "RWA-BD-002", "RWA-AR-001", "RWA-IN-001"],
        "資產類型": ["商辦大樓", "住宅地產", "企業債權", "基礎建設", "應收帳款", "綠能設施"],
        "初始估值 (USD)": [15000000, 3200000, 500000, 8000000, 1200000, 4500000],
        "個體波動率": [0.05, 0.08, 0.12, 0.03, 0.15, 0.06],
    }
    df = pd.DataFrame(data)

    # 模擬市場動態變化
    np.random.seed(int(time.time()))
    df['市場影響因子'] = np.random.normal(loc=0, scale=volatility, size=len(df))
    df['即時估值 (USD)'] = df['初始估值 (USD)'] * (1 + df['市場影響因子'])
    df['即時估值 (USD)'] = df['即時估值 (USD)'].astype(int)
    df['估值漲跌幅'] = (df['市場影響因子'] * 100).round(2).astype(str) + "%"

    # 模擬 MLP / Random Forest 預測出的違約機率 (Default Probability)
    df['AI預測違約機率'] = np.clip(df['個體波動率'] * 2 + df['市場影響因子'], 0.01, 0.99)

    # 風險評等轉換
    def get_rating(prob):
        if prob < 0.05: return "AAA"
        elif prob < 0.10: return "AA"
        elif prob < 0.15: return "A"
        elif prob < 0.25: return "BBB"
        else: return "C (高風險)"

    df['信用評等'] = df['AI預測違約機率'].apply(get_rating)
    df['AI預測違約機率'] = (df['AI預測違約機率'] * 100).round(2).astype(str) + "%"
    return df

df = load_and_predict_rwa_data(market_volatility)

# 5. KPI 數據卡片 (Metrics) - 讓首頁看起來更豐富
col1, col2, col3, col4 = st.columns(4)
col1.metric("追蹤資產總數", f"{len(df)} 筆")
col2.metric("總資產管理規模 (AUM)", f"${df['即時估值 (USD)'].sum():,}")
col3.metric("平均市場影響", f"{(df['市場影響因子'].mean()*100):.2f}%", delta=f"{(df['市場影響因子'].mean()*100):.2f}%")
col4.metric("高風險資產佔比", f"{(len(df[df['信用評等'] == 'C (高風險)']) / len(df) * 100):.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# 6. 使用 Tabs 進行版面收納與視覺化
tab1, tab2, tab3 = st.tabs(["📋 RWA 資產總覽", "📈 Matplotlib 數據視覺化", "💡 智能投資決策"])

with tab1:
    st.subheader("即時動態資產池 (Live Asset Pool)")
    # 使用 Pandas 客製化顏色顯示 dataframe
    st.dataframe(df[['資產代碼', '資產類型', '初始估值 (USD)', '即時估值 (USD)', '估值漲跌幅', '信用評等']], use_container_width=True)

with tab2:
    st.subheader("風險評級與資產分佈分析")
    col_chart1, col_chart2 = st.columns(2)

    # 運用 Matplotlib 繪製精美圓餅圖與長條圖
    with col_chart1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        rating_counts = df['信用評等'].value_counts()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax1.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.axis('equal')
        plt.title("RWA 信用評級分佈")
        st.pyplot(fig1)

    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df_sorted = df.sort_values(by="即時估值 (USD)", ascending=True)
        ax2.barh(df_sorted['資產類型'], df_sorted['即時估值 (USD)'] / 10000, color='#17becf')
        ax2.set_xlabel('即時估值 (萬 USD)')
        ax2.set_title("各資產類型估值排行")
        st.pyplot(fig2)

with tab3:
    st.subheader("個人化資產推薦引擎")
    if "保守" in risk_tolerance:
        recommendation = df[df['信用評等'].isin(['AAA', 'AA'])]
    elif "穩健" in risk_tolerance:
        recommendation = df[df['信用評等'].isin(['A', 'BBB'])]
    else:
        recommendation = df[df['信用評等'] == 'C (高風險)']

    if not recommendation.empty:
        st.success(f"✅ 根據您的設定，系統篩選出以下 {len(recommendation)} 筆適合的 RWA 標的：")
        st.table(recommendation[['資產代碼', '資產類型', '即時估值 (USD)', 'AI預測違約機率', '信用評等']])
    else:
        st.warning("⚠️ 目前市場環境下，無完全符合您風險偏好的標的，建議前往側邊欄調整參數。")
