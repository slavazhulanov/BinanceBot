import os
from binance.client import Client
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Константы
SYMBOL = 'MATICUSDT'  # Пара для торговли
TRADE_QUANTITY = 10   # Количество для торговли
API_KEY = 'your_api_key_here'
API_SECRET = 'your_api_secret_here'

# Создание клиента API
client = Client(API_KEY, API_SECRET)

# Получение свечных данных
klines = client.get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_15MINUTE, "2 day ago UTC") # Изменение интервала на 15 минут и получение данных за 2 дня

# Преобразование данных в DataFrame
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                                      'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Выбор необходимых столбцов
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = data[columns]

# Преобразование времени
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Нормализация данных
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Загрузка модели
model = tf.keras.models.load_model('MATICUSDT_RNN_model.h5')

# Получение последнего набора данных для прогнозирования
last_data = df.tail(model.input_shape[1])

# Преобразование данных для входа в RNN
last_data = np.reshape(last_data.values, (1, model.input_shape[1], 1))

# Прогнозирование
prediction = model.predict(last_data)

# Расчет относительного изменения цены
relative_change = (prediction[0][0] - last_data[0][-1][0]) / last_data[0][-1][0]

# Решение о торговле
if relative_change > 0.01:
    # Покупка
    print('BUY')
    order = client.create_order(
        symbol=SYMBOL,
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_MARKET,
        quantity=TRADE_QUANTITY
    )
    print(order)
    # Расчет профита
    trade_price = float(order['fills'][0]['price'])
    trade_quantity = float(order['executedQty'])
    profit = trade_price * trade_quantity * relative_change
    print(f"Profit: {profit}")
elif relative_change < -0.01:
    # Продажа
    print('SELL')
    order = client.create_order(
        symbol=SYMBOL,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_MARKET,
        quantity=TRADE_QUANTITY
    )
    print(order)
    # Расчет профита
    trade_price = float(order['fills'][0]['price'])
    trade_quantity = float(order['executedQty'])
    profit = trade_price * trade_quantity * relative_change
    print(f"Profit: {profit}")

