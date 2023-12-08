from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


"""
SimpleImputer is a scikit-learn class which is helpful in handling the missing data in the predictive model dataset. It replaces the NaN values with a specified placeholder. 

It is implemented by the use of the SimpleImputer() method which takes the following arguments :
1. missing_values : The missing_values placeholder which has to be imputed. By default is NaN 
2. strategy : The data which will replace the NaN values from the dataset. The strategy argument can take the values – ‘mean'(default), ‘median’, ‘most_frequent’ and ‘constant’. 
3. fill_value : The constant value to be given to the NaN data using the constant strategy. 
"""
from sklearn.impute import  SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
imputer = imputer.fit(data)
data = imputer.transform(data)

"""
Random Forest
parameters :
1. max_features : These are the maximum number of features Random Forest is allowed to try in individual tree [Auto/None, sqrt, 0.2/0.5(percentage of featues)]
2. n_estimators : This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.
3. min_sample_leaf : If you have built a decision tree before, you can appreciate the importance of minimum sample leaf size. Leaf is the end node of a decision tree. A smaller leaf makes the model more prone to capturing noise in train data.
4. n_jobs : This parameter tells the engine how many processors is it allowed to use. A value of “-1” means there is no restriction whereas a value of “1” means it can only use one processor
"""
from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor()

"""
DecisionTreeRegressor\DecisionTreeClassifier
parameters :
1. min_samples_split – Minimum number of samples a node must possess before splitting.
2. min_samples_leaf – Minimum number of samples a leaf node must possess.
3. min_weight_fraction_leaf – Minimum fraction of the sum total of weights required to be at a leaf node.
4. max_leaf_nodes – Maximum number of leaf nodes a decision tree can have.
5. max_features – Maximum number of features that are taken into the account for splitting each node.
"""
from sklearn.tree import DecisionTreeRegressor
DecisionTreeRegressor()
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier()