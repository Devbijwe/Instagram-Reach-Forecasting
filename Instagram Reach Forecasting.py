import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

data = pd.read_csv("Instagram-Reach.csv", encoding = 'latin-1')

data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day_name()


day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()


p, d, q = 8, 1, 2


model=sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
predictions = model.predict(len(data), len(data)+100)
print(predictions)