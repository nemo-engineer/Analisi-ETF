
import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objects as go
import gspread
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
import io

st.set_page_config(layout="wide")
st.title("📊 Analisi GARCH & Monte Carlo su ETF / Fondi")

# --- Caricamento dati ---
sorgente = st.radio("📂 Seleziona la sorgente dei dati:", ["CSV", "Excel", "Google Sheets"])

df = None
if sorgente == "CSV":
    file = st.file_uploader("Carica un file CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

elif sorgente == "Excel":
    file = st.file_uploader("Carica un file Excel", type=["xlsx"])
    if file:
        df = pd.read_excel(file)

else:
    sheet_url = st.text_input("🔗 Inserisci l'URL del Google Sheet")
    json_file = st.file_uploader("Carica le credenziali Google (JSON)", type=["json"])
    if sheet_url and json_file:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(json_file.name, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

if df is not None:
    st.subheader("👁️ Anteprima dei dati")
    st.dataframe(df.head())

    # --- Parametri di analisi ---
    col_data = st.selectbox("📅 Seleziona colonna delle date", df.columns)
    col_prezzo = st.selectbox("💶 Seleziona colonna dei prezzi", df.columns)
    distribuzione = st.selectbox("📈 Seleziona la distribuzione", ["normal", "t"])

    if st.button("▶️ Avvia analisi"):
        df[col_data] = pd.to_datetime(df[col_data])
        df = df.sort_values(by=col_data)
        df.set_index(col_data, inplace=True)
        returns = df[col_prezzo].pct_change().dropna()

        # Modelli da confrontare
        models = {
            "GARCH": arch_model(returns, vol="Garch", p=1, q=1, dist=distribuzione),
            "EGARCH": arch_model(returns, vol="EGarch", p=1, q=1, dist=distribuzione),
            "GJR-GARCH": arch_model(returns, vol="GARCH", p=1, o=1, q=1, dist=distribuzione)
        }

        best_model = None
        best_aic = np.inf
        results_dict = {}

        for name, model in models.items():
            res = model.fit(disp="off")
            results_dict[name] = res
            if res.aic < best_aic:
                best_aic = res.aic
                best_model = res

        st.success(f"📌 Modello selezionato automaticamente: **{best_model.model.volatility.__class__.__name__}**")
        st.write("AIC per ciascun modello:")
        for name, res in results_dict.items():
            st.write(f"{name}: AIC = {res.aic:.2f}")

        # Forecast
        horizon = st.slider("🔮 Giorni di forecast", 1, 20, 5)
        forecast = best_model.forecast(horizon=horizon)
        variances = forecast.variance.values[-1]
        st.subheader("📊 Forecast della volatilità")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=np.sqrt(variances), mode="lines+markers", name="Volatilità Prevista"))
        st.plotly_chart(fig)

        # Esporta risultati
        if st.checkbox("💾 Esporta forecast in Excel"):
            out_df = pd.DataFrame({"Giorni": list(range(1, horizon+1)), "Volatilità Prevista": np.sqrt(variances)})
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                out_df.to_excel(writer, index=False)
            st.download_button("📥 Scarica file Excel", output.getvalue(), file_name="forecast_volatilita.xlsx")
