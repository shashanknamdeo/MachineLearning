

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'C:\Users\ShirishPC\Downloads\iris.csv')
df.head(5)
df = df.drop(columns = ['Id'])
df.head(5)
df.info()
df.isnull().sum()


le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head(100)

X = df.drop(columns = ['Species'])
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

model = LogisticRegression() 
model.fit(X_train, Y_train)

print("Accuracy: ", model.score(X_test, Y_test) * 100)

# ------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r'C:\Users\ShirishPC\Downloads\iris.csv')
df.head(5)
df = df.drop(columns = ['Id'])
df.head(5)
df.info()
df.isnull().sum()


le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head(100)

X = df.drop(columns = ['Species'])
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, Y_train)

print("Accuracy: ", model.score(X_test, Y_test) * 100)


"""

Standardization process converts data to smaller values in the range 0 to 1 so that all of them lie on the same scale and one doesn’t overpower the other.

#Scaling numeric features using sklearn StandardScalar
numeric=['age', 'bmi', 'children']
sc=StandardScalar()
X_train[numeric]=sc.fit_transform(X_train[numeric])
X_test[numeric]=sc.transform(X_test[numeric])


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

df = iris.copy()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")



plt.boxplot(iris.data)
plt.show(block=False)


from sklearn.linear_model import LogisticRegression

array = iris.data

# x = array[:,0:4]
y = array[:,4]

validation_size = 0.2
seed = 6

x_test, x_train, y_test, y_train = model_selection.train_test_split(x, y, test_size=validation_size, rendom_state = seed)

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Loading dataset :

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ShirishPC\Downloads\iris.csv')
dataset.describe()

# Splitting the dataset into the Training set and Test set :

# Splitting the dataset into the Training set and Test set
x = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(x_train, y_train)


# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='auto',n_jobs=None, penalty='l2', random_state=0, solver='lbfgs',tol=0.0001, verbose=0, warm_start=False)

"""