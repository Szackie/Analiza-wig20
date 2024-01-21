#clean_data.py

import pandas as pd

# Wczytanie pliku CSV
data = pd.read_csv("wig20_all.csv")

#Sprawdzenie typów danych
print(data.dtypes)
data['Data'] = pd.to_datetime(data['Data'])
print(data.dtypes)

#Sprawdzenie braków danych
print(data.isnull().sum())

#data = data.drop(columns=['Brak danych','Wolumen','Najnizszy','Otwarcie','Najwyzszy'])

#Przekształcenie symboli spółek w kolumny
data_pivot = data.pivot(index='Data', columns='Symbol', values='Zamkniecie')
#data_pivot.to_csv('cleaned_data.csv')
#print(data_pivot)