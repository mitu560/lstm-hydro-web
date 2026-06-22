import streamlit as st
import pandas as pd
from model import run_lstm_pipeline, forecast_runoff, FEATURE_COLS
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 1. 字体路径
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Regular.otf")

# 2. 添加字体
font_manager.fontManager.addfont(font_path)

# 3. 获取字体的真实 family 名称（关键！）
prop = font_manager.FontProperties(fname=font_path)
real_font_name = prop.get_name()

# 4. 设置 matplotlib 全局字体
plt.rcParams["font.family"] = real_font_name
plt.rcParams["axes.unicode_minus"] = False

print("使用字体:", real_font_name)

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

uploaded_file = st.sidebar.file_uploader("📁 上传 Excel 文件", type=["xlsx"], key="history_file")
future_file = st.sidebar.file_uploader("🔮 上传未来逐日气象数据", type=["xlsx"], key="future_file")
win_size = st.sidebar.slider("⏳ 时间窗口", 5, 60, 12)
epochs = st.sidebar.slider("🔁 训练轮数", 20, 300, 120)
flood_q = st.sidebar.slider("🌊 洪水分位阈值", 0.70, 0.99, 0.85)

st.sidebar.markdown("---")
st.sidebar.info("☑ 上传历史数据后模型将自动训练；上传未来气象数据后可输出未来径流预测结果。")


# -------------------------------
# 主界面 Tabs（科技感 UI）
# -------------------------------
tab_train, tab_flood, tab_pred, tab_data, tab_forecast = st.tabs([
    "📈 模型训练结果",
    "🌊 洪水段分析",
    "📊 全序列预测",
    "📁 数据预览",
    "🔮 径流预测"
])


if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, parse_dates=["date"])

    # TAB 数据预览
    with tab_data:
        st.subheader("📁 历史训练数据预览")
        st.dataframe(df.head())

        st.subheader("📌 未来气象数据所需字段")
        st.write("未来气象 Excel 需要包含 `date` 和以下 14 个气象变量，不需要包含 `径流` 列。")
        st.code("date\n" + "\n".join(FEATURE_COLS), language="text")

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

    # TAB4：未来径流预测
    with tab_forecast:
        st.subheader("🔮 基于逐日气象输入的径流预测")
        st.write("上传未来逐日气象数据后，平台将基于当前训练完成的 LSTM 模型递推输出预测径流。")

        if future_file is not None:
            future_df = pd.read_excel(future_file, parse_dates=["date"])

            st.subheader("📁 未来气象数据预览")
            st.dataframe(future_df.head())

            if st.button("开始预测未来径流"):
                try:
                    pred_df = forecast_runoff(
                        results["model"],
                        results["scaler"],
                        results["history_df"],
                        future_df,
                        win_size
                    )

                    st.success("✅ 未来径流预测完成！")

                    st.subheader("📋 预测结果")
                    st.dataframe(pred_df)

                    fig_future = plt.figure(figsize=(10, 4))
                    plt.plot(pred_df["date"], pred_df["runoff_pred"], marker="o")
                    plt.grid(True)
                    plt.xlabel("日期")
                    plt.ylabel("预测径流")
                    plt.title("未来径流预测过程线")
                    plt.tight_layout()
                    st.pyplot(fig_future)

                    csv = pred_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 下载预测结果 CSV",
                        csv,
                        "future_runoff_prediction.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"预测失败：{e}")
        else:
            st.info("请在左侧上传未来逐日气象 Excel 文件。")

else:
    st.info("⬅ 请在左侧上传历史训练数据文件以开始运行模型。")
