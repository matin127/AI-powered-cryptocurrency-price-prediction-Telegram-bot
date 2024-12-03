import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#مرحله 1: دریافت داده‌ها از KuCoin با استفاده از CCXT
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data
#مرحله 2: پیش‌پردازش داده‌ها
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

#مرحله 3: ایجاد مدل LSTM
def build_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
#مرحله 4: آموزش مدل
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    return model
#مرحله 5: پیش‌بینی داده‌ها
def predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

#مرحله 6: نمایش داده‌ها
def plot_results(data, predictions):
    plt.figure(figsize=(16, 8))
    plt.plot(data, color='blue', label='Actual Price')
    plt.plot(predictions, color='red', label='Predicted Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#اجرای مراحل پروژه
if __name__ == "__main__":
    #دریافت داده‌ها از KuCoin
    symbol = "BTC/USDT"
    timeframe = "1h"  # بازه زمانی
    limit = 500       # تعداد کندل‌ها
    data = fetch_data(symbol, timeframe, limit)

    #استفاده از قیمت بسته شدن (close)
    close_prices = data['close']
    #تقسیم داده‌ها به بخش‌های آموزش و تست
    train_size = int(len(close_prices) * 0.8)
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    #پیش‌پردازش داده‌ها
    X_train, y_train, scaler = preprocess_data(train_data)
    X_test, y_test, _ = preprocess_data(test_data)
    
    #بازشکل‌دهی داده‌ها برای مدل
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #ساخت و آموزش مدل
    model = build_model()
    model = train_model(model, X_train, y_train)
    #پیش‌بینی داده‌ها
    predictions = predict(model, X_test, scaler)
    #نمایش نتایج
    plot_results(test_data.values, predictions)
