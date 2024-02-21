#prognoza.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Wczytaj dane
df = pd.read_csv("portfel.csv", index_col='Data', parse_dates=True)

# Definiuj parametry modelu
best_params = {
    'ALR': (0, 3, 1),
    'JSW': (5, 1, 0),
    'CDR': (3, 4, 2),
    'DNP': (1, 0, 0, 1, 0, 0, 12)
}

# Przygotuj słownik do przechowywania prognoz
predictions = {}

from pandas.tseries.offsets import DateOffset

# Trenuj model i generuj prognozy dla każdej kolumny
for column in df.columns:
    if column == 'DNP':
        model = SARIMAX(df[column], order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    else:
        model = ARIMA(df[column], order=best_params[column])
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # prognoza na następne 30 dni

    # Stwórz nowy indeks daty dla prognozy
    last_date = df.index[-1]
    new_index = pd.date_range(start=last_date + DateOffset(days=1), periods=30)
    forecast.index = new_index

    predictions[column] = forecast


future_data=pd.read_csv('cleaned_future_stocks.csv', index_col='Data', parse_dates=True)

# Wygeneruj wykresy prognoz
for column in predictions.keys():
    print('dane empiryczne:')
    print(future_data[column])
    print('prognozy:')
    print(predictions[column])
    plt.figure(figsize=(10,6))
    plt.plot(df[column], label='Historyczne')
    plt.plot(predictions[column], label='Prognozowane')
    plt.plot(future_data[column], label='Rzeczywiste')
    plt.title(f'{column} - Historyczne + Prognozowane')
    plt.legend()
    plt.show()