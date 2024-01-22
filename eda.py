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

weak_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i,j])<threshold:
            weak_corr.append((corr_matrix.columns[i],corr_matrix.columns[j],))


#ułóżmy te pary w kolejności od minimalnej korelacji, do maksymalnej ( dopuszczalnej przez próg). I wtedy dodawajmy pary po kolei, sprawdzając, czy korelacja tej spółki, którą chcemy dodać nie przekracza progu ze spółkami, które już dodaliśmy.

weak_corr = sorted(weak_corr, key=lambda x: abs(corr_matrix.loc[x[0], x[1]]))


independent_stocks = [weak_corr[0][0],weak_corr[0][1]]
print("pierwsza para?")
print(independent_stocks)

for pair in weak_corr:
    if pair[0] not in independent_stocks:
        for stock in independent_stocks:
            print("parka!!!")
            print([pair[0],stock])
            if (pair[0],stock) in weak_corr:
                independent_stocks.append(pair[0])
    if pair[1] not in independent_stocks:
        for stock in independent_stocks:
            if (pair[1],stock) in weak_corr:
                independent_stocks.append(pair[1])

print("Spółki nieskorelowane:")
for stock in independent_stocks:
    print(stock)
# 
# mask = np.ones(corr_matrix.shape, dtype=bool)
# for pair in weak_corr_pairs:
#     i = corr_matrix.columns.get_loc(pair[0])
#     j = corr_matrix.columns.get_loc(pair[1])
#     mask[i, j] = False
#     mask[j, i] = False

# cmap = colors.ListedColormap(['green' if mask[i, j] else 'red' for i in range(mask.shape[0]) for j in range(mask.shape[1])])


# sns.heatmap(corr_matrix, mask=mask, annot=True)
# plt.show()


# unique_stocks = list(set([stock for pair in weak_corr_pairs for stock in pair]))

# portfolio_stocks = []

# for stock in unique_stocks:
#     if all(abs(corr_matrix.loc[stock, portfolio_stock]) < threshold for portfolio_stock in portfolio_stocks):
#         portfolio_stocks.append(stock)

# print(portfolio_stocks)