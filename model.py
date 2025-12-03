import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 使用云端兼容的中文字体
font_manager.fontManager.addfont("fonts/NotoSansCJksc-Regular.otf")
plt.rcParams["font.family"] = "Noto Sans CJK"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


# ============ Hampel 滤波函数 ============
def hampel(vals_orig, k=7, t0=3):
    vals = vals_orig.values
    vals_filt = np.copy(vals)
    outliers_indices = []
    n = len(vals)

    for i in range(k, n - k):
        window = vals[i - k:i + k + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        if mad == 0:
            continue
        if np.abs(vals[i] - median) > t0 * mad:
            vals_filt[i] = median
            outliers_indices.append(i)
    return vals_filt, outliers_indices


# ============ 滑动窗口 ============
def prepare_data(data, win_size, target_feature_idx=0):
    X, y = [], []
    for i in range(len(data) - win_size):
        X.append(data[i:i + win_size, :])
        y.append(data[i + win_size, target_feature_idx])
    return np.asarray(X), np.asarray(y)


# ============ 主函数（训练 + 预测 + 绘图） ============
def run_lstm_pipeline(df, win_size=12, epochs=200, q_flood=0.85):

    # ============ 1. 基础字段 ============
    target_col = "径流"
    feature_cols = [
        'temperature_2m_C',
        'temperature_2m_max_C',
        'dewpoint_temperature_2m_C',
        'skin_temperature_C',
        'temperature_of_snow_layer_C',
        'snow_albedo_max_mm',
        'snow_cover_max_mm',
        'snow_density_max_mm',
        'snow_depth_max_mm',
        'snow_depth_water_equivalent_max_mm',
        'snowfall_sum_mm',
        'snowmelt_sum_mm',
        'total_precipitation_sum_mm',
        'total_evaporation_sum_mm'
    ]

    df = df.set_index("date")
    df["径流_raw"] = df["径流"]

    # Hampel 去异常
    df["径流"], _ = hampel(df["径流"])

    # ============ 2. 数据集 ============
    model_cols = [target_col] + feature_cols
    df_model = df[model_cols]

    train_size = int(len(df_model) * 0.8)
    df_train = df_model.iloc[:train_size]
    df_test = df_model.iloc[train_size:]

    # ============ 3. 归一化 ============
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    data_train = scaler.transform(df_train)
    data_test = scaler.transform(df_test)

    # ============ 4. 滑动窗口 ============
    train_x, train_y = prepare_data(data_train, win_size, 0)
    test_x, test_y = prepare_data(data_test, win_size, 0)

    # ============ 5. 洪水样本加权 ============
    runoff_train_real = df_train[target_col].values
    flood_threshold = np.quantile(runoff_train_real, q_flood)

    train_target_real = df_train[target_col].iloc[win_size:].values
    flood_mask_train = train_target_real > flood_threshold
    sample_weights = np.where(flood_mask_train, 5.0, 1.0)

    # ============ 6. LSTM 模型 ============
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        train_x, train_y,
        epochs=epochs,
        batch_size=32,
        validation_data=(test_x, test_y),
        sample_weight=sample_weights,
        verbose=0,
        callbacks=[es]
    )

    # ============ 7. 损失曲线图 ============
    fig_loss = plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.legend(); plt.grid(True)
    plt.title("训练损失曲线")

    # ============ 8. 反归一化 ============
    runoff_min = df_train[target_col].min()
    runoff_max = df_train[target_col].max()
    inv = lambda x: x * (runoff_max - runoff_min) + runoff_min

    y_pred = model.predict(test_x).ravel()
    y_pred_real = inv(y_pred)
    test_y_real = inv(test_y)

    # ============ 9. 指标 ============
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(test_y_real, y_pred_real))),
        "MAE": float(mean_absolute_error(test_y_real, y_pred_real)),
        "R2": float(r2_score(test_y_real, y_pred_real))
    }

    # ============ 10. 全序列预测图 ============
    train_pred = model.predict(train_x).ravel()
    train_pred_real = inv(train_pred)
    train_y_real = inv(train_y)

    train_dates = df_train.index[win_size:]
    test_dates = df_test.index[win_size:]

    fig_pred = plt.figure(figsize=(12, 5))
    plt.plot(train_dates, train_y_real, label="训练集真实", color="c")
    plt.plot(train_dates, train_pred_real, label="训练预测", color="b")
    plt.plot(test_dates, test_y_real, label="测试集真实", color="orange")
    plt.plot(test_dates, y_pred_real, label="测试预测", color="red")
    plt.legend(); plt.grid(True); plt.title("LSTM 预测效果")

    # ============ 11. 洪水段散点图 ============
    flood_mask_test = test_y_real > flood_threshold
    fig_flood = plt.figure(figsize=(6, 5))
    plt.scatter(test_y_real[flood_mask_test], y_pred_real[flood_mask_test], edgecolors='k')
    mn, mx = min(test_y_real.min(), y_pred_real.min()), max(test_y_real.max(), y_pred_real.max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.grid(True)
    plt.xlabel("观测径流"); plt.ylabel("模拟径流")
    plt.title("洪水段散点图")

    return {
        "metrics": metrics,
        "loss_fig": fig_loss,
        "pred_fig": fig_pred,
        "flood_fig": fig_flood
    }
