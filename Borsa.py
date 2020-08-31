import pandas as pd
import matplotlib.pyplot as plt 
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import numpy as np
#%%
df = pd.read_csv("USD_TRY Geçmiş Verileri.csv",header=0, index_col=0, parse_dates=True,
    squeeze=True)
#%%
for i in range(0,95):
    df["Şimdi"][i] = df["Şimdi"][i].replace(",",".")
#%%
df["Şimdi"] = df["Şimdi"].astype("float")    
#%%
# line plot
df["Şimdi"].plot()
pyplot.show()
#%%
# histogram plot
df["Şimdi"].hist()
pyplot.show()
#%% 
# autocorrelation
autocorrelation_plot(df["Şimdi"])
pyplot.show()
#%%
lag_plot(df["Şimdi"],lag=6)
#%%
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df["Şimdi"])
pyplot.show()
#%%
rolling = df["Şimdi"].rolling(window=5)
rolling_mean = rolling.mean()
rolling_mean.plot(color="red")
df["Şimdi"].plot(color="blue")
#%%
df2 = pd.concat([df["Şimdi"],rolling_mean],axis=1)
df2.columns = ["Simdi","MA"]
#%%
a=pd.Series(0)
a[0] = 0 
a[1] = 0
a[2] = 0 
a[3] = 0    
for i in range(4,95):
    a[i] = df2["Simdi"][i] - df2["MA"][i]
#%%
a.index = df2.index
#%%
df2 = pd.concat([df2,a],axis=1)
#%%
df2= df2[4:95]

#%%
es = pd.Series(0)
for i in range(4,95):
    es[i] =( 0.95*df["Şimdi"][i] + 0.95**2*df["Şimdi"][i-1] + 0.95**3*df["Şimdi"][i-2] + 0.95**4*df["Şimdi"][i-3] + + 0.95**5*df["Şimdi"][i-4])/5
#%%
y= df["Şimdi"][4:95]  
lag1 = df["Şimdi"][3:94]
lag2 = df["Şimdi"][2:93]
lag3 = df["Şimdi"][1:92]
lag4 = df["Şimdi"][0:91]
#%%
y.index = range(0,91)
lag1.index = range(0,91)
lag2.index = range(0,91)
lag3.index = range(0,91)
lag4.index = range(0,91)
#%%
df5= pd.concat([lag1,lag2,lag3,y],axis=1)
df5.columns = ["lag1","lag2","lag3","y"]
#%%
cor=df5.corr()
#%%
X = df5.drop("y",axis=1)
y=df5.y
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X= pca.fit_transform(X)
#%%
y = np.array(y).reshape(-1,1)
plt.scatter(X,y)
#%%

#%%
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
#%%
methods = ["LinearRegression","Lasso","Ridge","ElasticNet","PLSRegression","PLSSVD"]
algo = [LinearRegression(),Lasso(),Ridge(),ElasticNet()]
model = SVR("rbf").fit(X,y)
#%%
print(model.score(X,y))
#%%
#%%
df = pd.read_csv("USD_TRY Geçmiş Verileri kopyası.csv",header=0, index_col=0, parse_dates=True,
    squeeze=True)
#%%
for i in range(0,90):
    df["Şimdi"][i] = df["Şimdi"][i].replace(",",".")
#%%
df["Şimdi"] = df["Şimdi"].astype("float")  
#%%
y= df["Şimdi"][4:90]  
lag1 = df["Şimdi"][3:89]
lag2 = df["Şimdi"][2:88]
lag3 = df["Şimdi"][1:87]
lag4 = df["Şimdi"][0:86]
#%%
y.index = range(0,86)
lag1.index = range(0,86)
lag2.index = range(0,86)
lag3.index = range(0,86)
lag4.index = range(0,86)
#%%
df5= pd.concat([lag1,lag2,lag3,y],axis=1)
df5.columns = ["lag1","lag2","lag3","y"]
#%%
cor=df5.corr()
#%%
X_test = df5.drop("y",axis=1)
y_test=df5.y

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_test= pca.fit_transform(X_test)
#%%
y_test = np.array(y_test).reshape(-1,1)
plt.scatter(X_test,y_test)
#%%
y_pred = model.predict(X_test)
#%%
print(np.sqrt(mean_squared_error(y_test,y_pred))) 
#%%
pd.DataFrame(y_test).plot(color="red")
pd.DataFrame(y_pred).plot(color="blue")









  