import requests
import json
import pandas as pd

symbol = 'BTCUSDT'

url = 'https://api.binance.com/api/v3/klines'

params = {
    'symbol': symbol,
    'interval': '15m',
    'startTime': '1503038700000',  # начало данных в миллисекундах
    'endTime': '1681627382188',
    'limit': 1000,  # максимальное количество свечей за один запрос
}

headers = {
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
}

data = []

while True:
    response = requests.get(url, headers=headers, params=params)
    response_json = json.loads(response.text)
    data += response_json

    if len(response_json) < 1000:
        break

    params['startTime'] = response_json[-1][0] + 1

df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low',
                                 'close', 'volume', 'close_time', 'quote_asset_volume',
                                 'number_of_trades', 'taker_buy_base_asset_volume',
                                 'taker_buy_quote_asset_volume', 'ignore'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

df.to_csv('BTCUSDT.csv', index=False)
