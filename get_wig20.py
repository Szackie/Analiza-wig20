# get_wig20.py
import pandas as pd
import os

def download_data(symbol, start_date, end_date):
    url = f"https://stooq.pl/q/d/l/?s={symbol}&d1={start_date}&d2={end_date}&i=d"
    data = pd.read_csv(url)
    data['Symbol'] = symbol  # Dodajemy kolumnę z symbolem spółki
    return data
# Zahardkodowana lista symboli spółek
symbols = ['ALR', 'CCC', 'CDR', 'CPS', 'DNP', 'JSW', 'KGH', 'LTS', 'LPP', 'MBK', 'OPL', 'PEO', 'PGE', 'PGN', 'PKN', 'PKO', 'PLY', 'PZU', 'SPL', 'TPE']
# Pobranie danych dla każdej spółki
all_data = []
for symbol in symbols:
    data = download_data(symbol, "20230101", "20240115")
    all_data.append(data)

# Połączenie wszystkich danych w jedną ramkę danych
all_data = pd.concat(all_data)

# Zapis do pliku CSV:
filename = "wig20_all.csv"
all_data.to_csv(filename, index=False)

# Otwarcie pliku
os.startfile(filename)