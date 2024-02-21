# #model.py
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA

# import numpy as np

# df = pd.read_csv("portfel.csv")

# # Załóżmy, że chcemy użyć 80% danych do treningu i 20% do testowania
# train_size = 0.8

# # Oblicz liczbę wierszy w zestawie treningowym
# train_num = int(df.shape[0] * train_size)

# # Podziel dane na zestawy treningowe i testowe
# train_df = df.iloc[:train_num]
# test_df = df.iloc[train_num:]

# # Teraz `train_df` zawiera 80% początkowych wierszy danych, a `test_df` zawiera pozostałe 20%

# print(train_df)
# print(test_df)

# # Słownik do przechowywania prognoz dla każdej kolumny
# predictions = {}
# tests = {}
# min_mse={}



# # for column in train_df.columns[1:]:
# #     y_train = train_df[column]

# #     model = ARIMA(y_train, order=(5,1,0))

# #     model_fit = model.fit()

# #     y_pred = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)

# #     tests[column]= test_df[column]
# #     # Zapisz prognozy dla tej kolumny
# #     predictions[column] = y_pred
# #     mse = mean_squared_error(tests[column],predictions[column])
# #     print(f'MSE {column}: {mse}')

# # MSE ALR: 84.3302185432102
# # MSE JSW: 11.843689481087306  -- może być
# # MSE CDR: 46.18485366067129
# # MSE DNP: 3740.621479499256
# min_mse['JSW']=11.843689481087306
# best_params={}
# # lowest_mse={}

# best_params['JSW'] = (5,1,0)
# #     lowest_mse['JSW'] = 11.843689481087306


# #dla reszty dobieram parametry metodą grid search
    
# # columns = [col for col in train_df.columns[1:] if col !='JSW']
# # print(columns)
# # p_range = range(0,6)
# # d_range = range(0,6)
# # q_range = range(0,6)

# # for column in columns:
# #     y_train = train_df[column]
# #     y_test = test_df[column]

# #     best_params[column] = None
# #     lowest_mse[column] = np.inf

# #     for p in p_range:
# #         for d in d_range:
# #             for q in q_range:
# #                 try:
# #                     model = ARIMA(y_train, order=(p,d,q))
# #                     model_fit = model.fit()

# #                     y_pred = model_fit.predict(start=len(train_df),end=len(train_df)+len(test_df)-1)

# #                     mse = mean_squared_error(y_test, y_pred)

# #                     if mse < lowest_mse[column]:
# #                         best_params[column] = (p, d, q)
# #                         lowest_mse[column] = mse
# #                 except:
# #                     continue
# # for col in columns:
# #     print(f'Best params for {col}: {best_params[col]}, MSE: {lowest_mse[col]}')



# # Best params for ALR: (0, 3, 1), MSE: 6.557770249093277  --OK
# # Best params for CDR: (3, 4, 2), MSE: 10.408574870811051 --OK
# # Best params for DNP: (4, 3, 4), MSE: 1677.697466214512

# best_params['ALR']= (0,3,1)
# best_params['CDR']= (3,4,2)
# min_mse['ALR'] = 6.557770249093277
# min_mse['CDR'] = 10.408574870811051

# # from statsmodels.tsa.statespace.sarimax import SARIMAX

# # Definiujemy zakresy dla parametrów p, d, q oraz P, D, Q, s
# # p_range = range(0, 3)
# # d_range = range(0, 3)
# # q_range = range(0, 3)
# # P_range = range(0, 3)
# # D_range = range(0, 3)
# # Q_range = range(0, 3)
# # s_range = [0, 12]  # zakładamy, że sezonowość może być miesięczna lub brak

# # y_train = train_df['DNP']
# # y_test = test_df['DNP']

# # Zapiszemy najlepsze parametry i najniższy MSE
# # best_params = None
# # lowest_mse = np.inf

# # Przeszukujemy siatkę parametrów
# # for p in p_range:
# #     for d in d_range:
# #         for q in q_range:
# #             for P in P_range:
# #                 for D in D_range:
# #                     for Q in Q_range:
# #                         for s in s_range:
# #                             try:
# #                                 model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s))
# #                                 model_fit = model.fit()

# #                                 y_pred = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)

# #                                 mse = mean_squared_error(y_test, y_pred)

# #                                 # Jeśli MSE jest niższe niż dotychczas najniższe, zapiszemy te parametry
# #                                 if mse < lowest_mse:
# #                                     best_params = (p, d, q, P, D, Q, s)
# #                                     lowest_mse = mse

# #                             except:
# #                                 continue

# # print(f'Best params for DNP: {best_params}, MSE: {lowest_mse}')


# #At iterate   50    f=  3.26758D+00    |proj g|=  5.02591D-04

# #            * * *

# # Tit   = total number of iterations
# # Tnf   = total number of function evaluations
# # Tnint = total number of segments explored during Cauchy searches
# # Skip  = number of BFGS updates skipped
# # Nact  = number of active bounds at final generalized Cauchy point
# # Projg = norm of the final projected gradient
# # F     = final function value

# #            * * *

# #    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
# #     9     50     63      1     0     0   5.026D-04   3.268D+00
# #   F =   3.26758311806408

# # STOP: TOTAL NO. of ITERATIONS REACHED LIMIT
# # C:\Users\Szymon\anaconda3\Lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
# #   warnings.warn("Maximum Likelihood optimization failed to "


# #Best params for DNP: (1, 0, 0, 1, 0, 0, 12), MSE: 329.1053547512303
# best_params['DNP'] = (1, 0, 0, 1, 0, 0, 12)
# min_mse['DNP'] = 329.1053547512303
# #Sprawdzam wariancję, żeby ocenić MSE
# variance={}
# for column in train_df.columns[1:]:
#     variance[column]= train_df[column].var()
#     print(f'Wariancja dla kolumny {column}: {variance[column]}, MSE: {min_mse[column]}, BEST PARAMS: {best_params[column]}')

# # # Wariancja dla kolumny ALR: 51.0976731155779, MSE: 6.557770249093277, BEST PARAMS: (0, 3, 1)
# # # Wariancja dla kolumny JSW: 56.38201414824117, MSE: 11.843689481087306, BEST PARAMS: (5, 1, 0)
# # # Wariancja dla kolumny CDR: 341.07862969846747, MSE: 10.408574870811051, BEST PARAMS: (3, 4, 2)
# # # Wariancja dla kolumny DNP: 1510.9146530150758, MSE: 329.1053547512303, BEST PARAMS: (1, 0, 0, 1, 0, 0, 12)

# wykresy porównujące dane testowe z prognozowanymi

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

df = pd.read_csv("portfel.csv")

train_size = 0.8
train_num = int(df.shape[0] * train_size)
train_df = df.iloc[:train_num]
test_df = df.iloc[train_num:]

best_params = {}

best_params['ALR']=(0, 3, 1)
best_params['JSW']= (5, 1, 0)
best_params['CDR']= (3, 4, 2)
best_params['DNP']= (1, 0, 0, 1, 0, 0, 12)

predictions = {}
tests = {}

for column in train_df.columns[1:-1]:
    print(column)
    y_train = train_df[column]

    model = ARIMA(y_train, order=best_params[column])

    model_fit = model.fit()

    y_pred = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)

    tests[column]= test_df[column]
    # Zapisz prognozy dla tej kolumny
    predictions[column] = y_pred

y_train = train_df['DNP']
model = SARIMAX(y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
model_fit=model.fit()
y_pred = model_fit.predict(start=len(train_df),end=len(train_df) + len(test_df)-1)
tests['DNP'] = test_df['DNP']
predictions['DNP'] = y_pred

for column in tests.keys():
    plt.figure(figsize=(10,6))
    plt.plot(tests[column], label='Ceny akcji')
    plt.plot(predictions[column], label='Model')
    plt.title(f'{column} - Porównanie na zbiorze testowym')
    plt.legend()
    plt.show()

# spróbuję teraz przewidzieć przyszłe ceny akcji na podstawie modelów. zobaczymy czy opłacałoby się inwestować (zgodnie z modelem Markovitza)

