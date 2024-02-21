#eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv("cleaned_data.csv")

data.hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()

numeric_data = data.select_dtypes(include=[np.number])

corr_matrix = numeric_data.corr()
threshold = 0.5

# Definiowanie własnej mapy kolorów
colors = ["black", "green", "black"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Tworzenie mapy ciepła
sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0)

# Wyświetlanie mapy ciepła
plt.show()


weak_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i,j])<threshold:
            weak_corr.append((corr_matrix.columns[i],corr_matrix.columns[j]))

selected_stock = set()

for stock1, stock2 in weak_corr:
    if all(abs(corr_matrix.loc[stock, stock1]) < threshold and abs(corr_matrix.loc[stock, stock2]) < threshold for stock in selected_stock):
        selected_stock.add(stock1)
        selected_stock.add(stock2)

print("Spółki nieskorelowane:")
print(selected_stock)

# Utwórz nowy DataFrame zawierający tylko wybrane kolumny
selected_stock_df = data[["Data"]+list(selected_stock)]

# Zapisz nowy DataFrame do pliku CSV
selected_stock_df.to_csv("portfel.csv", index=False)

for name in selected_stock_df.columns[1:]:
    print(f"Przeprowadzam analizę dla spółki {name}")
   
    print(data[name].describe())
