#markovitz.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_return(weights, returns):
    return np.sum(returns.mean()*weights)*252

def calculate_portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

def minimize_volatility(weights, returns):
    return calculate_portfolio_volatility(weights, returns)

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(minimize_volatility, num_assets*[1./num_assets,],args=args, method ='SLSQP', bounds = bounds, constraints=constraints)
    return result

df_first_row = pd.read_csv("portfel.csv", nrows=1)
column_names = df_first_row.columns.tolist()
col_to_read= column_names[1:]
returns=pd.read_csv("portfel.csv", usecols=col_to_read)

result = optimize_portfolio(returns)
optimized_weights = result.x

print(f'Optimized weights: {optimized_weights}')
print(f'Expected return: {calculate_portfolio_return(optimized_weights,returns)}')
print(f'Volatility: {calculate_portfolio_volatility(optimized_weights,returns)}')

# Optimized weights: [0.25195446 0.61946166 0.11481541 0.01376846]
# Expected return: 15549.369706644462
# Volatility: 65.99517959114633  

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
portfolio_return = calculate_portfolio_return(optimized_weights, returns)
portfolio_volatility = calculate_portfolio_volatility(optimized_weights, returns)
ax.scatter(portfolio_volatility, portfolio_return, color='blue')
ax.set_title('Optymalizacja portfela')
ax.set_xlabel('Volatility (std)')
ax.set_ylabel('Expected Returns')
ax.annotate('Optimized Portfolio', (portfolio_volatility, portfolio_return), xytext=(10, 10), 
            textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.show()