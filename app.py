import streamlit as st
import pandas as pd
from model import run_lstm_pipeline
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -------------------------------
# 全局中文字体
# -------------------------------
font_manager.fontManager.addfont("fonts/NotoSansCJK-Regular.otf")
plt.rcParams["font.family"] = "Noto Sans CJK"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 页面配置（科技蓝）
# -------------------------------
st.set_page_config(
    page_title="LSTM 洪水预测平台",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------
# 顶部蓝色科技风横幅
# -------------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0A84FF, #005BBB);
    padding: 25px; border-radius: 12px;
    text-align: center;">
    <h1 style="color:white; font-size:36px; margin-bottom:0;">
        🌊 LSTM 洪水预测平台
    </h1>
    <p style="color:white; font-size:18px;">
        高流量识别 · 洪水模拟 · 序列预测 · 云端实时运行
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# Sidebar（现代科技蓝 UI）
# -------------------------------
st.sidebar.markdown("""
## ⚙ 参数设置
请选择模型配置来开始预测。
""")

uploaded_file = st.sidebar.file_uploader("📁 上传 Excel 文件", type=["xlsx"])
win_size = st.sidebar.slider("⏳ 时间窗口", 5, 60, 12)
epochs = st.sidebar.slider("🔁 训练轮数", 20, 300, 120)
flood_q = st.sidebar.slider("🌊 洪水分位阈值", 0.70, 0.99, 0.85)

st.sidebar.markdown("---")
st.sidebar.info("☑ 上传数据后模型将自动训练。")


# -------------------------------
# 主界面 Tabs（科技感 UI）
# -------------------------------
tab_train, tab_flood, tab_pred, tab_data = st.tabs([
    "📈 模型训练结果",
    "🌊 洪水段分析",
    "📊 全序列预测",
    "📁 数据预览"
])


if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, parse_dates=["date"])

    # TAB 数据预览
    with tab_data:
        st.subheader("📁 数据预览")
        st.dataframe(df.head())

    # 模型训练
    with st.spinner("🚀 正在训练模型，请稍候…"):
        results = run_lstm_pipeline(df, win_size=win_size, epochs=epochs, q_flood=flood_q)

    st.success("🎉 模型训练完成！")

    # TAB1：训练结果
    with tab_train:
        st.subheader("📉 损失曲线")
        st.pyplot(results["loss_fig"])

        st.subheader("📌 模型评价指标（卡片风格）")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{results['metrics']['RMSE']:.3f}")
        col2.metric("MAE", f"{results['metrics']['MAE']:.3f}")
        col3.metric("R²", f"{results['metrics']['R2']:.3f}")

    # TAB2：洪水段分析
    with tab_flood:
        st.subheader("🌊 洪水段拟合效果")
        st.pyplot(results["flood_fig"])

    # TAB3：完整序列预测
    with tab_pred:
        st.subheader("📊 训练 + 测试 + 未来预测")
        st.pyplot(results["pred_fig"])

else:
    st.info("⬅ 请在左侧上传文件以开始运行模型。")
