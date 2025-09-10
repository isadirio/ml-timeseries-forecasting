import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Time Series Forecast", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename_axis("Date").reset_index()
    return df

def add_features(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["return"] = out[target_col].pct_change()
    # Lag features
    for lag in [1, 5, 10, 20]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
        out[f"ret_lag_{lag}"] = out["return"].shift(lag)
    # Rolling stats
    for win in [5, 20, 60]:
        out[f"roll_mean_{win}"] = out[target_col].rolling(win).mean()
        out[f"roll_std_{win}"] = out[target_col].rolling(win).std()
    # Calendar
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out = out.dropna().reset_index(drop=True)
    return out

def train_test_split_by_horizon(df: pd.DataFrame, horizon: int):
    return df.iloc[:-horizon].copy(), df.iloc[-horizon:].copy()

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}

def naive_forecast(train: pd.Series, horizon: int):
    return np.repeat(train.iloc[-1], horizon)

def moving_average_forecast(train: pd.Series, horizon: int, window: int = 5):
    avg = train.iloc[-window:].mean()
    return np.repeat(avg, horizon)

def holt_winters_forecast(train_series: pd.Series, horizon: int):
    model = ExponentialSmoothing(
        train_series, trend="add", seasonal=None, initialization_method="estimated"
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(horizon)
    return fc.values

def rf_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col="Close"):
    features = [c for c in train_df.columns if c not in ["Date", target_col]]
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return preds, rf

def main():
    st.title("📈 Financial Time Series Forecast")
    st.caption("Mini app demo: baseline vs modelli semplici su dati Yahoo Finance")

    with st.sidebar:
        st.header("Impostazioni")
        ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
        today = date.today()
        start = st.date_input("Start", value=today - timedelta(days=365*3))
        end = st.date_input("End", value=today)
        horizon = st.slider("Horizon di test (giorni)", min_value=7, max_value=90, value=30, step=1)
        baseline_window = st.slider("Finestra Moving Average", 3, 30, 5, 1)
        model_choice = st.multiselect(
            "Modelli da confrontare",
            ["Naive (last value)", "Moving Average", "Holt-Winters", "Random Forest"],
            default=["Naive (last value)", "Moving Average", "Holt-Winters", "Random Forest"]
        )
        st.markdown("---")
        st.caption("Suggerimenti: AAPL, MSFT, TSLA, BTC-USD, ^GSPC")

    if not ticker:
        st.warning("Inserisci un ticker valido.")
        st.stop()

    with st.spinner("Scarico dati..."):
        df = load_data(ticker, str(start), str(end))

    if df.empty:
        st.error("Nessun dato trovato per il ticker/periodo selezionati.")
        st.stop()

    st.subheader(f"Dati grezzi — {ticker}")
    st.write(df.tail())

    # Plot close price
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.set_title(f"{ticker} — Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # Feature engineering per RF
    fe_df = add_features(df[["Date", "Close"]].copy(), target_col="Close")

    if len(fe_df) <= horizon + 30:
        st.warning("Serie troppo corta per l'horizon scelto. Riduci l'horizon o estendi il periodo.")
        st.stop()

    train_df, test_df = train_test_split_by_horizon(fe_df, horizon)
    y_train = train_df["Close"]
    y_test = test_df["Close"]
    test_dates = test_df["Date"]

    results = {}
    rf_model = None

    if "Naive (last value)" in model_choice:
        y_pred = naive_forecast(y_train, horizon)
        results["Naive (last value)"] = (y_pred, evaluate(y_test.values, y_pred))

    if "Moving Average" in model_choice:
        y_pred = moving_average_forecast(y_train, horizon, window=baseline_window)
        results["Moving Average"] = (y_pred, evaluate(y_test.values, y_pred))

    if "Holt-Winters" in model_choice:
        try:
            y_pred = holt_winters_forecast(y_train, horizon)
            results["Holt-Winters"] = (y_pred, evaluate(y_test.values, y_pred))
        except Exception as e:
            st.warning(f"Holt-Winters non disponibile: {e}")

    if "Random Forest" in model_choice:
        try:
            y_pred_rf, rf_model = rf_forecast(train_df, test_df, target_col="Close")
            results["Random Forest"] = (y_pred_rf, evaluate(y_test.values, y_pred_rf))
            # Feature importances
            fi = pd.Series(
                rf_model.feature_importances_,
                index=[c for c in train_df.columns if c not in ["Date", "Close"]]
            ).sort_values(ascending=False).head(15)
            st.subheader("Importanza delle feature (Random Forest)")
            st.bar_chart(fi)
        except Exception as e:
            st.warning(f"Random Forest non disponibile: {e}")

    if results:
        metrics_df = pd.DataFrame({name: met for name, (_, met) in results.items()}).T
        st.subheader("Metriche su test set")
        st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE%": "{:.2f}", "R2": "{:.3f}"}))

        st.subheader("Confronto previsioni")
        fig2, ax2 = plt.subplots()
        ax2.plot(test_dates, y_test.values, label="Reale")
        for name, (preds, _) in results.items():
            ax2.plot(test_dates, preds, label=name)
        ax2.set_title("Forecast vs Reale")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

        best_name = min(results.items(), key=lambda kv: kv[1][1]["RMSE"])[0]
        best_pred = results[best_name][0]
        residuals = y_test.values - best_pred
        st.subheader(f"Analisi residui — modello migliore: {best_name}")
        fig3, ax3 = plt.subplots()
        ax3.plot(test_dates, residuals, label="Residui")
        ax3.axhline(0, linestyle="--", linewidth=1)
        ax3.set_title("Residui nel tempo")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Errore")
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    st.markdown("---")
    st.caption("Demo. Le previsioni non sono consigli finanziari.")

if __name__ == "__main__":
    main()
