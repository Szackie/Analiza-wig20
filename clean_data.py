#clean_data.py
import pandas as pd

# Wczytanie pliku CSV
data = pd.read_csv("wig20_all.csv")

#Sprawdzenie typów danych
print(data.dtypes)
data['Data'] = pd.to_datetime(data['Data'])
print(data.dtypes)
