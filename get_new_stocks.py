# get_new_stocks.py
import pandas as pd
import os

def download_data(symbol, start_date, end_date):
    url = f"https://stooq.pl/q/d/l/?s={symbol}&d1={start_date}&d2={end_date}&i=d"
    data = pd.read_csv(url)
    data['Symbol'] = symbol  # Dodajemy kolumnę z symbolem spółki
    return data
# Zahardkodowana lista symboli spółek
symbols = ['ALR', 'CDR', 'DNP', 'JSW']
# Pobranie danych dla każdej spółki
all_data = []
for symbol in symbols:
    data = download_data(symbol, "20231230", "20240130")
    all_data.append(data)

# Połączenie wszystkich danych w jedną ramkę danych
all_data = pd.concat(all_data)

# Zapis do pliku CSV:
filename = "future_stock.csv"
all_data.to_csv(filename, index=False)

# Otwarcie pliku
#os.startfile(filename)


# Wczytanie pliku CSV
data = pd.read_csv("future_stock.csv")

#Sprawdzenie typów danych
print(data.dtypes)
data['Data'] = pd.to_datetime(data['Data'])
print(data.dtypes)

#Sprawdzenie braków danych
print(data.isnull().sum())

data = data.drop(columns=['Wolumen','Najnizszy','Otwarcie','Najwyzszy'])

#Przekształcenie symboli spółek w kolumny
data_pivot = data.pivot(index='Data', columns='Symbol', values='Zamkniecie')
data_pivot.to_csv('cleaned_future_stocks.csv')
print(data_pivot)