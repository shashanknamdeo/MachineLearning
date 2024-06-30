import pandas as pd
import seaborn as sb

data_path = r'C:\Users\ShashankPC\Downloads\HousePricesART\train.csv'
df = pd.read_csv(data_path)
print(df)
sb.histplot(data=df['MSSubClass'])

df["A"].astype("category")



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Example dataset (replace with your own data)
# Load your dataset here
# For example: df = pd.read_csv('your_dataset.csv')

# Generate a random dataset for demonstration purposes
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regressor": SVR(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Gradient Boosting Regressor" : GradientBoostingRegressor()
}

# Function to evaluate the model
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:\n\tMean Squared Error: {mse}\n\tR^2 Score: {r2}\n")

# Apply all models
for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)


Linear Regression:
    Mean Squared Error: 0.024147876979174258
    R^2 Score: 0.8301641508417135

Ridge Regression:
    Mean Squared Error: 0.024045794174516168
    R^2 Score: 0.8308821153993646

Lasso Regression:
    Mean Squared Error: 0.031086169858088788
    R^2 Score: 0.7813660364643875

Decision Tree:
    Mean Squared Error: 0.04719090320338008
    R^2 Score: 0.6680988922958052

Random Forest:
    Mean Squared Error: 0.024388897143976365
    R^2 Score: 0.8284690177917672

Support Vector Regressor:
    Mean Squared Error: 0.03878719552659932
    R^2 Score: 0.727203501392289

K-Neighbors Regressor:
    Mean Squared Error: 0.048015833802997905
    R^2 Score: 0.6622970245372617

Gradient Boosting Regressor:
    Mean Squared Error: 0.019607310431546756
    R^2 Score: 0.8620986756010158


Id                  0
MSSubClass          0
MSZoning            0
LotFrontage       259
LotArea             0
Street              0
Alley            1369
LotShape            0
LandContour         0
Utilities           0
LotConfig           0
LandSlope           0
Neighborhood        0
Condition1          0
Condition2          0
BldgType            0
HouseStyle          0
OverallQual         0
OverallCond         0
YearBuilt           0
YearRemodAdd        0
RoofStyle           0
RoofMatl            0
Exterior1st         0
Exterior2nd         0
MasVnrType        872
MasVnrArea          8
ExterQual           0
ExterCond           0
Foundation          0
BsmtQual           37
BsmtCond           37
BsmtExposure       38
BsmtFinType1       37
BsmtFinSF1          0
BsmtFinType2       38
BsmtFinSF2          0
BsmtUnfSF           0
TotalBsmtSF         0
Heating             0
HeatingQC           0
CentralAir          0
Electrical          1
1stFlrSF            0
2ndFlrSF            0
LowQualFinSF        0
GrLivArea           0
BsmtFullBath        0
BsmtHalfBath        0
FullBath            0
HalfBath            0
BedroomAbvGr        0
KitchenAbvGr        0
KitchenQual         0
TotRmsAbvGrd        0
Functional          0
Fireplaces          0
FireplaceQu       690
GarageType         81
GarageYrBlt        81
GarageFinish       81
GarageCars          0
GarageArea          0
GarageQual         81
GarageCond         81
PavedDrive          0
WoodDeckSF          0
OpenPorchSF         0
EnclosedPorch       0
3SsnPorch           0
ScreenPorch         0
PoolArea            0
PoolQC           1453
Fence            1179
MiscFeature      1406
MiscVal             0
MoSold              0
YrSold              0
SaleType            0
SaleCondition       0
SalePrice           0

"""
column for review
ScreenPorch

"""

dataDF_dtnumber = dataDF.select_dtypes(include=np.number)
dataDF_dtobject = dataDF.select_dtypes(include=object)
dataDF_dtnumber.shape, dataDF_dtobject.shape

label_encoder_LotShape = LabelEncoder()
dataDF_dtobject_labelencoded = dataDF_dtobject.apply(lambda x: LabelEncoder().fit_transform(x))
dataDF_dtobject_labelencoded.shape

dataDF_combined = pd.concat([dataDF_dtnumber, dataDF_dtobject_labelencoded], axis=1)

# pd.set_option('display.max_rows', None)
# print(dataDF_combined)
dataDF_combined_not_nan = dataDF_combined.drop(['LotFrontage' ,  'GarageYrBlt', 'MasVnrArea'] , axis=1)
dataDF_combined_not_nan.isna().sum().sum()

X = dataDF_combined_not_nan.drop('SalePrice', axis=1)
y = dataDF_combined_not_nan['SalePrice']
y = y.apply(lambda x: (math.log(x)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# GradientBoostingRegressor
Mean Squared Error: 0.02
0.8955177524006276

estimator = DecisionTreeRegressor()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Counter(y_pred)

estimator = RandomForestRegressor()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

estimator = GradientBoostingRegressor()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
r2_score(y_true=y_test, y_pred=y_pred)

estimator = LinearRegression()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
r2_score(y_true=y_test, y_pred=y_pred)

pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

estimator.coef_

estimator = Lasso()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
r2_score(y_true=y_test, y_pred=y_pred)

estimator.coef_

estimator = Ridge()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print('r2_score:', r2_score(y_test, y_pred))

pd.DataFrame([y_test])

pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

dataDF1 = dataDF.drop(['Alley', 'MiscFeature', 'PoolQC'], axis=1)

Counter(dataDF['MiscVal'])