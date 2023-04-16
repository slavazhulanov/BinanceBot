import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

# Загрузка данных из CSV-файла
df = pd.read_csv('MATICUSDT.csv')

# Преобразование времени
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].apply(lambda x: pd.Timestamp(x).strftime('%s'))

# Выбор необходимых столбцов
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = df[columns]

# Нормализация данных
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df.drop('close', axis=1), df['close'], test_size=0.25, shuffle=False)

# Создание модели RNN
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Преобразование данных для входа в RNN
X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

# Коллбэк для сохранения только лучших эпох
checkpoint_filepath = 'MATICUSDT_RNN_model_{epoch:04d}_{val_loss:.8f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Обучение модели
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback])

# График обучения
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('loss.png')
