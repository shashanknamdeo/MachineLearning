"""
The medv variable is the target variable.

crim : per capita crime rate by town.
zn : proportion of residential land zoned for lots over 25,000 sq.ft.
indus : proportion of non-retail business acres per town.
chas : Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
nox : nitrogen oxides concentration (parts per 10 million).
rm : average number of rooms per dwelling.
age : proportion of owner-occupied units built prior to 1940.
dis : weighted mean of distances to five Boston employment centres.
rad : index of accessibility to radial highways.
tax : full-value property-tax rate per $10,000.
ptratio : pupil-teacher ratio by town.
black : 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
lstat : lower status of the population (percent).
medv : median value of owner-occupied homes in $1000s.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xg
import lightgbm as lgbm
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# file_path = r'C:\Users\ShashankPC\Downloads\boston-housing\test.csv'
# test_df = pd.read_csv(file_path)


file_path = r'C:\Users\ShashankPC\Downloads\boston-housing\train.csv'
df = pd.read_csv(file_path)

df = df.drop(['ID'], axis=1)

df.describe()
#              crim          zn       indus        chas         nox          rm         age         dis         rad         tax     ptratio       black       lstat        medv
# count  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000  333.000000
# mean     3.360341   10.689189   11.293483    0.060060    0.557144    6.265619   68.226426    3.709934    9.633634  409.279279   18.448048  359.466096   12.515435   22.768769
# std      7.352272   22.674762    6.998123    0.237956    0.114955    0.703952   28.133344    1.981123    8.742174  170.841988    2.151821   86.584567    7.067781    9.173468
# min      0.006320    0.000000    0.740000    0.000000    0.385000    3.561000    6.000000    1.129600    1.000000  188.000000   12.600000    3.500000    1.730000    5.000000
# 25%      0.078960    0.000000    5.130000    0.000000    0.453000    5.884000   45.400000    2.122400    4.000000  279.000000   17.400000  376.730000    7.180000   17.400000
# 50%      0.261690    0.000000    9.900000    0.000000    0.538000    6.202000   76.700000    3.092300    5.000000  330.000000   19.000000  392.050000   10.970000   21.600000
# 75%      3.678220   12.500000   18.100000    0.000000    0.631000    6.595000   93.800000    5.116700   24.000000  666.000000   20.200000  396.240000   16.420000   25.000000
# max     73.534100  100.000000   27.740000    1.000000    0.871000    8.725000  100.000000   10.710300   24.000000  711.000000   21.200000  396.900000   37.970000   50.000000

df.isnull().sum().sort_values(ascending=False)/df.shape[0]
# crim       0.0
# zn         0.0
# indus      0.0
# chas       0.0
# nox        0.0
# rm         0.0
# age        0.0
# dis        0.0
# rad        0.0
# tax        0.0
# ptratio    0.0
# black      0.0
# lstat      0.0
# medv       0.0
# dtype: float64

# sns.pairplot(data=df)
# plt.show(block=False)

# fig, ax = plt.subplots(4,4, figsize=(16, 16))
# sns.boxplot(ax=ax[0][0], data=df.crim)
# sns.boxplot(ax=ax[0][1], data=df.zn)
# sns.boxplot(ax=ax[0][2], data=df.indus)
# sns.boxplot(ax=ax[0][3], data=df.chas)
# sns.boxplot(ax=ax[1][0], data=df.nox)
# sns.boxplot(ax=ax[1][1], data=df.rm)
# sns.boxplot(ax=ax[1][2], data=df.age)
# sns.boxplot(ax=ax[1][3], data=df.dis)
# sns.boxplot(ax=ax[2][0], data=df.rad)
# sns.boxplot(ax=ax[2][1], data=df.tax)
# sns.boxplot(ax=ax[2][2], data=df.ptratio)
# sns.boxplot(ax=ax[2][3], data=df.black)
# sns.boxplot(ax=ax[3][0], data=df.lstat)
# plt.show(block=False)

# sns.boxplot(data=df.crim)

# sns.heatmap(data=df.corr(), annot=True)
# plt.show(block=False)

# y = df.medv.values
# x = df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# numeric = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
# sc = StandardScaler()
# x_train[numeric] = sc.fit_transform(x_train[numeric])
# x_test[numeric] = sc.transform(x_test[numeric])

# model = LinearRegression()
# model.fit(x_train, y_train)
# yhat = model.predict(x_test)
# r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), np.sqrt(mean_squared_error(y_test, yhat))



# def boost_models(x):
#     regr_trans = TransformedTargetRegressor(regressor=x, transformer=QuantileTransformer(output_distribution='normal'))
#     regr_trans.fit(x_train, y_train)
#     yhat = regr_trans.predict(x_test)
#     algoname= x.__class__.__name__
#     return algoname, round(r2_score(y_test, yhat),3), round(mean_absolute_error(y_test, yhat),1), round(np.sqrt(mean_squared_error(y_test, yhat)),1)

# algo = [LinearRegression(), RandomForestRegressor(), SVR(), Lasso(), GradientBoostingRegressor(), lgbm.LGBMRegressor(), xg.XGBRFRegressor()]
# score=[]

# for a in algo :
#     score.append(boost_models(a))

# pd.DataFrame(score, columns=['Model', 'Score', 'MAE', 'RMSE'])


# def fitModels(modal):
#     modal = modal.fit(x_train, y_train)
#     yhat = modal.predict(x_test)
#     algoname= modal.__class__.__name__
#     return algoname, round(r2_score(y_test, yhat),3), round(mean_absolute_error(y_test, yhat),1), round(np.sqrt(mean_squared_error(y_test, yhat)),1)

# algo = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), SVR(), Lasso(), GradientBoostingRegressor(), lgbm.LGBMRegressor(), xg.XGBRFRegressor()]
# score=[]

# for a in algo :
#     score.append(fitModels(modal=a))

# print(pd.DataFrame(score, columns=['Model', 'Score', 'MAE', 'RMSE']))

#                        Model  Score  MAE  RMSE
# 0           LinearRegression  0.739  3.6   4.8
# 1      DecisionTreeRegressor  0.730  2.9   4.9
# 2      RandomForestRegressor  0.910  2.1   2.8
# 3                        SVR  0.574  3.6   6.2
# 4                      Lasso  0.704  3.7   5.2
# 5  GradientBoostingRegressor  0.947  1.7   2.2
# 6              LGBMRegressor  0.919  2.1   2.7
# 7             XGBRFRegressor  0.875  2.2   3.4

# from sklearn.model_selection import GridSearchCV

# param_grid = { 
#     'n_estimators': [150, 200, 250], 
#     'max_features': [ None], 
#     'max_depth': [9, 12, 15], 
#     'max_leaf_nodes': [9, 12, 15], 
# } 

# grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid) 
# grid_search.fit(x_train, y_train) 
# print(grid_search.best_estimator_) 

# modal = RandomForestRegressor(n_estimators=1000).fit(x_train, y_train)
# yhat = modal.predict(x_test)
# r2_score(y_test, yhat)


# param_grid = { 
#     'n_estimators': [100, 200, 300], 
#     'max_features': ['sqrt',40,60], 
#     'max_depth'   : [4,6,8]
# } 

# grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid) 
# grid_search.fit(x_train, y_train) 
# print(grid_search.best_estimator_) 

# modal = GradientBoostingRegressor(max_depth=4, max_features=60).fit(x_train, y_train)
# yhat = modal.predict(x_test)
# r2_score(y_test, yhat)


# file_path = r'C:\Users\ShashankPC\Downloads\boston-housing\test.csv'
# df1 = pd.read_csv(file_path)
# y = df1.medv.values
# x = df1[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]