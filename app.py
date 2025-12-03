import streamlit as st
import pandas as pd
from model import run_lstm_pipeline

st.set_page_config(page_title="径流洪水预测平台", layout="wide")

st.title("🌊 基于 LSTM 的径流与洪水预测平台")

st.write("""
本平台用于：**径流预测、洪水段敏感性拟合、未来 10 日预测、损失曲线分析**。
请上传包含 `date`、径流、气象因子的 Excel 文件。
""")

uploaded_file = st.file_uploader("📤 上传 AKS_LSTM.xlsx 文件（需包含 date 列）", type=["xlsx"])

# 设置参数
win_size = st.slider("时间窗口长度（win_size）", 5, 60, 12)
epochs = st.slider("训练轮数（epochs）", 50, 500, 200)
flood_q = st.slider("洪水分位阈值（q）", 0.7, 0.99, 0.85)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, parse_dates=["date"])
    st.write("### 📌 数据表预览")
    st.dataframe(df.head())

    st.info("⏳ 正在训练 LSTM 模型，请稍等...")
    results = run_lstm_pipeline(df, win_size=win_size, epochs=epochs, q_flood=flood_q)
    st.success("🎉 模型训练完成！")

    # 显示损失图
    st.write("### 📉 训练损失曲线")
    st.pyplot(results["loss_fig"])

    # 显示预测图
    st.write("### 📈 训练集 & 测试集 & 未来预测")
    st.pyplot(results["pred_fig"])

    # 洪水段
    st.write("### 🌊 洪水段模拟效果")
    st.pyplot(results["flood_fig"])

    # 指标
    st.write("### 📑 模型评价指标")
    st.json(results["metrics"])

    st.success("平台运行完毕 ✔")

