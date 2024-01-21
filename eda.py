#eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors

data = pd.read_csv("cleaned_data.csv")

# for column in data.columns[1:]:
#     print(f"Przeprowadzam analizę dla spółki {column}")
   
#     print(data[column].describe())

# data.hist(bins=50, figsize=(15,10))
# plt.show()

numeric_data = data.select_dtypes(include=[np.number])

corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True)
# plt.show()

threshold = 0.5

weak_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i,j])<threshold:
            weak_corr_pairs.append((corr_matrix.columns[i],corr_matrix.columns[j]))


# for pair in weak_corr_pairs:
#     print(pair)

mask = np.ones(corr_matrix.shape, dtype=bool)
for pair in weak_corr_pairs:
    i = corr_matrix.columns.get_loc(pair[0])
    j = corr_matrix.columns.get_loc(pair[1])
    mask[i, j] = False
    mask[j, i] = False

cmap = colors.ListedColormap(['green' if mask[i, j] else 'red' for i in range(mask.shape[0]) for j in range(mask.shape[1])])

sns.heatmap(corr_matrix, mask=~mask, cmap=cmap, annot=True)
# plt.show()
sns.heatmap(corr_matrix, mask=mask, annot=True)
# plt.show()


unique_stocks = list(set([stock for pair in weak_corr_pairs for stock in pair]))

portfolio_stocks = []

for stock in unique_stocks:
    if all(abs(corr_matrix.loc[stock, portfolio_stock]) < threshold for portfolio_stock in portfolio_stocks):
        portfolio_stocks.append(stock)

print(portfolio_stocks)