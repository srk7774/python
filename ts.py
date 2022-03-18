import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/srk7774/data/master/retail_sales.csv',parse_dates=True,index_col='DATE')
ax = df['Sales'].plot(figsize = (12,6))
ax.autoscale(axis = 'both',tight = True)
ax.set(ylabel='Liquor Store Retail Sales(M)',xlabel="Dates",title='US Liquor Retail Sales Data (in Millions USD)');

from statsmodels.tsa.seasonal import seasonal_decompose
result_add = seasonal_decompose(df['Sales'],model = 'add')
result_add.plot();

result_mul = seasonal_decompose(df['Sales'],model = 'mul')
result_mul.plot();


df['Sales_6M_SMA'] = df['Sales'].rolling(window=6).mean()
df['Sales_12M_SMA'] = df['Sales'].rolling(window=12).mean()
df[['Sales','Sales_6M_SMA','Sales_12M_SMA']].plot(figsize=(18,6))

df['EWMA_12'] = df['Sales'].ewm(span=12,adjust=False).mean()
df[['Sales','EWMA_12']].plot(figsize = (18,6))

from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Setting the index frequency directly to monthly start, thus statsmodels does not need to infer it.
df.index.freq = 'MS'
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df['DES_12_add'] = ExponentialSmoothing(df['Sales'],trend='add').fit().fittedvalues.shift(-1)
df['DES_12_mul'] = ExponentialSmoothing(df['Sales'],trend='mul').fit().fittedvalues.shift(-1)
df['TES_12_add'] = ExponentialSmoothing(df['Sales'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df['TES_12_mul'] = ExponentialSmoothing(df['Sales'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
df[['Sales','DES_12_add','DES_12_mul','TES_12_mul','TES_12_add']].iloc[-128:].plot(figsize=(12,6)).autoscale(axis='x',tight=True)
