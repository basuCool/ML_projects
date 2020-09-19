
file = 'C:\builds\ML\Exercises\Life_expectancy\12603_17232_bundle_archive\Life_Expectancy_Data.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''df = pd.read_csv('Life_Expectancy_Data.csv')
df.isnull().sum()

# Fill Na with mean values
df.fillna(df.mean(), inplace=True)

# name of csv file  
filename = 'WHO.csv'
df.to_csv('WHO.csv', index=False)'''
df2 = pd.read_csv('WHO.csv')
'''
['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']
'''
sns.heatmap(df2.corr())
sns.pairplot(df2)
X = df2.drop('Life expectancy ', axis=1)
X = X.drop('Measles ', axis=1)
X = X.drop('infant deaths', axis=1)
X = X.drop('Population', axis=1)
X = X.drop('Year', axis=1)

y = df2['Life expectancy ']

# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X.iloc[:,0] = labelEncoder_X.fit_transform(X.iloc[:, 0])
X.iloc[:,1] = labelEncoder_X.fit_transform(X.iloc[:, 1])
#One hot encoding for categorical features
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Avoding the dummy variable trap by ignoring the one column from onehotencoded X
X = X[:, 1:]

#One hot encoding for categorical features
onehotencoder = OneHotEncoder(categorical_features = [192])
X = onehotencoder.fit_transform(X).toarray()
# Avoding the dummy variable trap by ignoring the one column from onehotencoded X
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''y_train = y_train.reshape(len(y_train), 1)
y_train = sc_y.fit_transform(y)'''

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train,y_train)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

from sklearn.linear_model import Ridge
regressor = Ridge(alpha=1.0)
regressor.fit(X_train,y_train)

from sklearn.linear_model import Lasso
regressor = Lasso(alpha=0.01)
regressor.fit(X_train,y_train)

from sklearn.linear_model import ElasticNet
regressor = ElasticNet(alpha = 0.075)
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

from math import sqrt
sqrt(mean_squared_error(y_test, y_pred))

'''
LinearRegression (with all features) 
r2_score = -3.9167736440430055e+21, MSE =  3.4950211024032886e+23
LinearRegression (without infant deaths, Year, Population, Measles):
r2_score = -2.910981472382996e+21, MSE =  2.5975311823692052e+23

SVR(with all features) r2_score = 0.8892, MSE = 9.887624575511383
SVR(without infant deaths, Year, Population, Measles) r2_score = 0.8892, MSE = 9.887624575511383
SVR(without Population, Measles) r2_score = 0.89, MSE = 9.88, RMSE = 3.12
Ridge regresion (without infant deaths, Year, Population, Measles) r2_score = 0.945, RMSE = 2.20
'''
'''from sklearn.decomposition import PCA 
pca = PCA(0.90) 
  
X_train_Pca = pca.fit_transform(X_train) 
X_test_Pca = pca.transform(X_test) 
 
pca = PCA(0.90)
pca.fit(X_train)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
 
explained_variance = pca.explained_variance_ratio_ '''









