import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import locale

# Set locale untuk parsing bulan dalam Bahasa Indonesia
locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Inflasi BI", layout="wide")
st.title("📊 Prediksi Inflasi Indonesia (2003-2024) dengan LSTM")

# 1. Load dan Preprocessing Data
@st.cache_data
def load_data():
    try:
        # Baca file Excel
        df = pd.read_excel("data_inflasi_bi_2003_2024.xlsx", sheet_name="Data Inflasi", header=4, usecols=["Periode", "Data Inflasi"])
        
        # Cleaning data
        df = df.rename(columns={"Data Inflasi": "Inflasi"})
        df = df.dropna()
        
        # Konversi nilai inflasi ke float
        df['Inflasi'] = df['Inflasi'].str.replace(' %', '').str.replace(',', '.').astype(float)
        
        # Fungsi untuk konversi periode ke datetime
        def convert_periode(periode):
            try:
                bulan_tahun = periode.split()
                bulan = bulan_tahun[0].capitalize()
                tahun = bulan_tahun[1] if len(bulan_tahun) > 1 else "2024"
                return datetime.strptime(f"{bulan} {tahun}", "%B %Y")
            except:
                return pd.NaT
        
        df['Tanggal'] = df['Periode'].apply(convert_periode)
        df = df.dropna().sort_values('Tanggal').set_index('Tanggal')
        return df[['Inflasi']]
    
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

data = load_data()
if data is None:
    st.stop()

# Tampilkan data
st.subheader("Data Inflasi BI (2003-2024)")
st.dataframe(data.head())

# 2. Visualisasi Data
st.subheader("Trend Inflasi Bulanan")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data.index, data['Inflasi'], label='Inflasi (%)', color='blue')
ax.set_xlabel("Tahun")
ax.set_ylabel("Inflasi (%)")
ax.grid(True)
st.pyplot(fig)

# 3. Preprocessing Data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Inflasi']])

# Fungsi untuk membuat dataset LSTM
def create_dataset(data, window_size=12):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = st.sidebar.slider("Pilih Window Size:", 6, 24, 12)
X, y = create_dataset(data_scaled, window_size)

# Bagi data (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Bangun Model LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training model
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training model LSTM..."):
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        st.success("Model selesai ditraining!")
        
        # Plot loss
        fig_loss = plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        st.pyplot(fig_loss)

# 5. Prediksi dan Evaluasi
if st.sidebar.button("🔮 Prediksi"):
    y_pred = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred)
    
    # Hitung RMSE
    rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual)**2))
    st.metric("RMSE", f"{rmse:.4f}")
    
    # Plot hasil prediksi
    fig_pred = plt.figure(figsize=(12, 4))
    plt.plot(data.index[-len(y_test):], y_test_actual, label='Aktual', color='blue')
    plt.plot(data.index[-len(y_test):], y_pred_actual, label='Prediksi', color='red', linestyle='--')
    plt.legend()
    st.pyplot(fig_pred)

# 6. Prediksi Masa Depan
st.sidebar.subheader("Prediksi ke Depan")
n_future = st.sidebar.number_input("Jumlah Bulan:", 1, 12, 6)

if st.sidebar.button("🌐 Generate Prediksi"):
    last_window = data_scaled[-window_size:]
    future_preds = []
    for _ in range(n_future):
        next_pred = model.predict(last_window.reshape(1, window_size, 1))
        future_preds.append(next_pred[0, 0])
        last_window = np.append(last_window[1:], next_pred)
    
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(
        start=data.index[-1] + pd.DateOffset(months=1),
        periods=n_future,
        freq='MS'
    )
    
    fig_future = plt.figure(figsize=(12, 4))
    plt.plot(data.index[-12:], data['Inflasi'][-12:], label='Data Historis', color='blue')
    plt.plot(future_dates, future_preds, label='Prediksi', color='green', marker='o')
    plt.legend()
    st.pyplot(fig_future)
