import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Cargar datos con manejo de errores
DATA_PATH = "most_streamed_spotify_songs_2024.csv"
try:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
except Exception as e:
    st.error(f"No se pudo cargar el archivo: {e}")
    st.stop()

# Limpiar nombres de columnas eliminando espacios adicionales
df.rename(columns=lambda x: x.strip(), inplace=True)

# Verificar existencia de la columna correcta
if "streams" not in df.columns:
    st.error("Error: La columna 'streams' no existe en el archivo CSV.")
    st.stop()

# Convertir valores de streams a numéricos y filtrar valores menores a 1000
df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
df = df[df["streams"] >= 1000]  # Filtramos datos irrelevantes

# Transformación logarítmica
df["log_streams"] = np.log10(df["streams"])

# Estimación de distribución normal sobre log_streams
mu, sigma = df["log_streams"].mean(), df["log_streams"].std()

# --- Parámetros ya calculados sobre log10(streams) ---
mu, sigma = df["log_streams"].mean(), df["log_streams"].std()

def simulacion_monte_carlo(n_simulaciones):
    Z = np.random.normal(loc=mu, scale=sigma, size=n_simulaciones)
    X = 10 ** Z
    return X

# Configuración de Streamlit
st.set_page_config(page_title="Simulación de Streams", layout="wide")
st.title("🎵 Simulación de Streams en Spotify")

# Sidebar para configuración de simulación
st.sidebar.header("Configuración")
n_simulaciones = st.sidebar.slider("Número de simulaciones", 10, 10000, step=1)
growth_factor    = st.sidebar.slider("Factor de Crecimiento (% aumento)", 0, 100, 30)

# El slider de umbral a un rango más realista
umbral = st.sidebar.slider(
    "Umbral de éxito (reproducciones)",
    min_value=1_000_000,
    max_value=1_000_000_000,
    value=10_000_000,
    step=100
)
st.sidebar.write(f"🎯 Umbral seleccionado: {umbral:,} reproducciones")

# Simulación de datos
simulated_streams = simulacion_monte_carlo(n_simulaciones)
adjusted_streams  = simulated_streams * (1 + growth_factor / 100)

# Probabilidades antes y después de marketing
prob_sin = np.mean(simulated_streams >= umbral)
prob_con = np.mean(adjusted_streams  >= umbral)
# Probabilidad de superar el umbral con el efecto de marketing
prob = np.mean(adjusted_streams >= umbral)
st.subheader("🎯 Probabilidad de Éxito")
st.write(f"Probabilidad de superar **{umbral:,}** streams después de marketing: **{prob:.4f}** ({prob*100:.2f}%)")
st.write(f"- Sin marketing: **{prob_sin:.4f}** ({prob_sin*100:.2f}%)")
st.write(f"- Con  marketing: **{prob_con:.4f}** ({prob_con*100:.2f}%)")
# Simulación de datos
simulated_streams = simulacion_monte_carlo(n_simulaciones)
adjusted_streams = simulated_streams * (1 + growth_factor / 100)

# Histograma simulado original
st.subheader("📉 Distribución Simulada de Streams (sin marketing)")
fig_sim = px.histogram(
    pd.DataFrame({"Simulados": simulated_streams}),
    x="Simulados", nbins=50,
    title="Distribución Simulada de Streams",
    color_discrete_sequence=["orange"]
)
st.plotly_chart(fig_sim)

# Histograma tras marketing
st.subheader("📈 Streams después de Marketing")
fig_growth = px.histogram(
    pd.DataFrame({"Streams Ajustados": adjusted_streams}),
    x="Streams Ajustados", nbins=50,
    title="Streams después de campaña de marketing",
    color_discrete_sequence=["green"]
)
st.plotly_chart(fig_growth)
