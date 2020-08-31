import pandas as pd
import matplotlib.pyplot as plt 
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import numpy as np
#%%
series = pd.read_csv("daily-total-female-births-CA.csv",header=0, index_col=0, parse_dates=True,
    squeeze=True)
print(type(series))
print(series.head())
#%%
print(series.size)
print(series["1959-01"])
print(series.describe())
#%%matplotlib.auto
series.plot()
#%%matplotlib.auto
series.plot(style='k.')
#%% 
series.hist()
#%%
series.plot(kind="kde")
#%%
lag_plot(series)
#%%
autocorrelation_plot(series)
#%%
series = pd.read_csv("international-airline-passengers.csv",header=0, index_col=0, parse_dates=True,
    squeeze=True)
series.plot()
#%%
series = series[0:144]
#%%
autocorrelation_plot(series)
#%% Square Root Transform-- Trendi azlaltır
df = pd.DataFrame(series.values)
df["SQT"] = np.sqrt(df.iloc[:,0])
#%%
df["SQT"].plot()
autocorrelation_plot(df["SQT"])
#%% Log Transform
df["LT"] = np.log(df.iloc[:,0])
#%%
df["LT"].plot()
autocorrelation_plot(df["LT"])
#%% Box-Cox Transform
from scipy.stats import boxcox
df["BX"] = boxcox(df.iloc[:,0], lmbda=0.0)
#%%
df["BX"].hist()
#%%
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
rolling_mean.plot(color="red")
series.plot(color="blue")
#%% Temporal Structure
#%% White Noise
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)
#%%
# summary stats
print(series.describe())
#%%
# line plot
series.plot()
pyplot.show()
#%%
# histogram plot
series.hist()
pyplot.show()
#%% 
# autocorrelation
autocorrelation_plot(series)
pyplot.show()
#%%
lag_plot(series)
#%%
# create and plot a random series
from random import seed
from random import randrange
from matplotlib import pyplot
seed(1)
series = [randrange(10) for i in range(1000)]
plt.plot(series)
#%%
pd.Series(series).hist()

#%% 
# autocorrelation
autocorrelation_plot(series)
pyplot.show()
#%%
lag_plot(pd.Series(series))
#%%
# create and plot a random walk
from random import seed
from random import random
from matplotlib import pyplot
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
  movement = -1 if random() < 0.5 else 1
  value = random_walk[i-1] + movement
  random_walk.append(value)
pyplot.plot(random_walk)
pyplot.show()
#%%
# autocorrelation
autocorrelation_plot(random_walk)
pyplot.show()
#%%
lag_plot(pd.Series(random_walk))
#%% Dickey fuller test
# calculate the stationarity of a random walk
from random import seed
from random import random
from statsmodels.tsa.stattools import adfuller
# generate random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
  movement = -1 if random() < 0.5 else 1
  value = random_walk[i-1] + movement
  random_walk.append(value)
# statistical test
result = adfuller(random_walk)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#%%
# persistence forecasts for a random walk
from random import seed
from random import random
from sklearn.metrics import mean_squared_error
from math import sqrt
# generate the random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
# persistence
predictions = list()
history = train[-1]
for i in range(len(test)):
  yhat = history
  predictions.append(yhat)
  history = test[i]
rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)
#%%
# random predictions for a random walk
from random import seed
from random import random
from sklearn.metrics import mean_squared_error
from math import sqrt
# generate the random walk
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
  movement = -1 if random() < 0.5 else 1
  value = random_walk[i-1] + movement
  random_walk.append(value)
# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
# random prediction
predictions = list()
history = train[-1]
for i in range(len(test)):
  yhat = history + (-1 if random() < 0.5 else 1)
  predictions.append(yhat)
  history = test[i]
rmse = sqrt(mean_squared_error(test, predictions))
print('Random RMSE: %.3f' % rmse)
#%% Time Series Decomposition-Additive Model
from statsmodels.tsa.seasonal import seasonal_decompose
series = pd.read_csv("daily-total-female-births-CA.csv",header=0, index_col=0, parse_dates=True,
    squeeze=True)
result = seasonal_decompose(series, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
#%%
result.plot()
#%% Time Series Decomposition-Multiplicative Model
result = seasonal_decompose(series, model='multiplicative',freq=1)
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
#%%
result.plot()
#%%
# calculate stationarity test of time series data
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#%% Evaluate models
# calculate a train-test split of a time series dataset
from pandas import read_csv
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True) 
X = series.values
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
#%%
# plot train-test split of time series data
plt.plot(train[:,1])
plt.plot([None for i in train[:,1]] + [x for x in test[:,1]])
plt.show()
#%%
# calculate forecast error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))] 
print('Forecast Errors: %s' % forecast_errors)
#%%
# calculate mean forecast error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))] 
bias = sum(forecast_errors) * 1.0/len(expected)
print('Bias: %f' % bias)
#%%
# calculate mean absolute error
from sklearn.metrics import mean_absolute_error 
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mae = mean_absolute_error(expected, predictions) 
print('MAE: %f' % mae)
#%%
# calculate mean squared error
from sklearn.metrics import mean_squared_error 
expected = [0.0, 0.5, 0.0, 0.5, 0.0] 
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mse = mean_squared_error(expected, predictions) 
print('MSE: %f' % mse)
#%%
# calculate root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mse = mean_squared_error(expected, predictions) 
rmse = sqrt(mse)
print('RMSE: %f' % rmse)
#%%
# Create lagged dataset
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1) 
dataframe.columns = ['t', 't+1'] 
print(dataframe.head(5))
#%%
# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
#%%
# persistence model
def model_persistence(x):
  return x
#%%
# walk-forward validation
from sklearn.metrics import mean_squared_error
predictions = list()
for x in test_X:
  yhat = model_persistence(x)
  predictions.append(yhat)
rmse = np.sqrt(mean_squared_error(test_y, predictions)) 
print('Test RMSE: %.3f' % rmse)
#%%
# evaluate a persistence forecast model
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1) 
dataframe.columns = ['t', 't+1'] 
print(dataframe.head(5))
# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
def model_persistence(x):
  return x
# walk-forward validation
predictions = list()
for x in test_X:
  yhat = model_persistence(x)
  predictions.append(yhat)
rmse = sqrt(mean_squared_error(test_y, predictions)) 
print('Test RMSE: %.3f' % rmse)
# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y]) 
pyplot.plot([None for i in train_y] + [x for x in predictions]) 
pyplot.show()
#%% Forecast Models
# lag plot of time series
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
lag_plot(series)
#%%
# correlation of lag=1
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1) 
dataframe.columns = ['t', 't+1']
result = dataframe.corr()
print(result)
#%%
# autocorrelation plot of time series
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series[0:144])
pyplot.show()
#%%
# autocorrelation plot of time series
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(series[0:144], lags=31)
pyplot.show()
#%%
series = series[0:144]
from statsmodels.tsa.arima_model import ARIMA
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid) 
residuals.plot()
plt.show()
# density plot of residuals 
residuals.plot(kind='kde') 
pyplot.show()
# summary stats of residuals
print(residuals.describe())

#%% Arima Forecasting
# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
      model = ARIMA(history, order=(5,1,0))
      model_fit = model.fit(disp=0)
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      obs = test[t]
      history.append(obs)
      print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions)) 
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes 
pyplot.plot(test)
pyplot.plot(predictions, color='red') 
pyplot.show()
#%%
# ACF plot of time series
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series)
pyplot.show()
#%%
# PACF plot of time series
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=50)
pyplot.show()
#%%
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
#%%
# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
  train_size = int(len(X) * 0.66)
  train, test = X[0:train_size], X[train_size:]
  history = [x for x in train]
  # make predictions
  predictions = list()
  for t in range(len(test)):
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test[t])
  # calculate out of sample error
  rmse = sqrt(mean_squared_error(test, predictions))
  return rmse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values): dataset = dataset.astype('float32')
best_score, best_cfg = float("inf"), None
for p in p_values:
    for d in d_values:
     for q in q_values:
       order = (p,d,q)
       try:
         rmse = evaluate_arima_model(dataset, order)
         if rmse < best_score:
             best_score, best_cfg = rmse, order 
         print('ARIMA%s RMSE=%.3f' % (order,rmse))
       except:
         continue
     print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
# load dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
















