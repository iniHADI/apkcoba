import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Inflasi BI", layout="wide")
st.title("📊 Prediksi Inflasi Indonesia (2003-2024) dengan LSTM")
st.markdown("""
Aplikasi ini menggunakan model **LSTM** untuk memprediksi inflasi bulanan berdasarkan data resmi Bank Indonesia.
""")

# 1. Load Data (Sesuai struktur Excel: Periode & Inflasi)
@st.cache_data
def load_data():
    try:
        # Baca file Excel
        data = pd.read_excel(
            "data_inflasi_bi_2003_2024.xlsx",
            parse_dates=['Periode'],  # Gunakan kolom 'Periode' sebagai tanggal
            usecols=['Periode', 'Inflasi'],  # Hanya baca kolom yang diperlukan
            engine='openpyxl'
        )
        
        # Rename kolom untuk konsistensi
        data = data.rename(columns={
            'Periode': 'Tanggal',
            'Inflasi': 'Inflasi_MoM'
        })
        
        data.set_index('Tanggal', inplace=True)
        return data
    
    except Exception as e:
        st.error(f"Gagal memuat data: {e}\nPastikan file Excel memiliki kolom 'Periode' dan 'Inflasi'.")
        return None

data = load_data()
if data is None:
    st.stop()  # Hentikan aplikasi jika data tidak valid

# Tampilkan data
st.subheader("Data Historis Inflasi BI (2003-2024)")
st.dataframe(data.head())

# 2. Visualisasi Data
st.subheader("Trend Inflasi Bulanan (MoM)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data.index, data['Inflasi_MoM'], label='Inflasi MoM', color='blue', linewidth=1)
ax.set_xlabel("Tahun")
ax.set_ylabel("Inflasi (%)")
ax.grid(True)
st.pyplot(fig)

# 3. Preprocessing Data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Inflasi_MoM']])

# Fungsi untuk membuat dataset LSTM
def create_dataset(data, window_size=12):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = st.sidebar.slider("**Pilih Window Size:**", 6, 24, 12)
X, y = create_dataset(data_scaled, window_size)

# Bagi data (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Bangun Model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Training model
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training model LSTM..."):
        history = model.fit(
            X_train, y_train, 
            epochs=100, 
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
    plt.plot(future_dates, future_preds, label='Prediksi', color='green', marker='o')
    plt.legend()
    st.pyplot(fig_future)
