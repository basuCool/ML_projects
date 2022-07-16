#Step 1: Importing the Relevant Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#Step 2: Data Inspection
train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
train.shape,test.shape

#ratio of null values
train.isnull().sum()/train.shape[0] *100

#ratio of null values
test.isnull().sum()/test.shape[0] *100

#categorical features
categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

#numerical features
numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])

#categorical features
categorical_test = test.select_dtypes(include =[np.object])
print("Categorical Features in Test Set:",categorical.shape[1])

#numerical features
numerical_test= test.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Test Set:",numerical.shape[1])

#Step 3: Data Cleaning

train.isnull().sum()
test.isnull().sum()

plt.figure(figsize=(8,5))
sns.boxplot('Item_Weight',data=train)

plt.figure(figsize=(8,5))
sns.boxplot('Item_Weight',data=test)

# Imputing with Mean
train['Item_Weight']= train['Item_Weight'].fillna(train['Item_Weight'].mean())
test['Item_Weight']= test['Item_Weight'].fillna(test['Item_Weight'].mean())

train['Outlet_Size'].isnull().sum(),test['Outlet_Size'].isnull().sum()

print(train['Outlet_Size'].value_counts())
print('******************************************')
print(test['Outlet_Size'].value_counts())

#Imputing with Mode
train['Outlet_Size']= train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
test['Outlet_Size']= test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])

#Step 4: Exploratory Data Analysis
train.columns
train.head()

#Item_Identifier

plt.figure(figsize=(25,7))
sns.countplot('Item_Identifier',data=train,palette='spring')

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Item_Identifier", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

"""
The function below was created specifically for the one-way ANOVA table results returned for Type II sum of squares
"""

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

k = anova_table(aov_table)

import scipy.stats as stats

stats.shapiro(model.resid)

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()
# Item_Identifier is significant with total sales.

#Item_Weight
sns.scatterplot(x= 'Item_Weight', y='Item_Outlet_Sales', hue = 'Item_Identifier',data=train)

# Item_Weight is having 0.011550 correlation with Item_Outlet_Sales. So this feature should be dropped

#Item_Fat_Content
train['Item_Fat_Content'].value_counts()

train['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)

train['Item_Fat_Content']= train['Item_Fat_Content'].astype(str)

plt.figure(figsize=(8,5))
sns.countplot('Item_Fat_Content',data=train,palette='ocean')

sns.countplot('Item_Outlet_Sales',data=train,hue ='Item_Fat_Content', palette='ocean')

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Item_Fat_Content", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.08. Null hypothesis of both these varaibles are independent to each other cannot be rejected. Should drop this feture.
import scipy.stats as stats

stats.shapiro(model.resid)

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Item_Visibility

plt.figure(figsize=(8,5))
sns.boxplot('Item_Visibility',data=train)

sns.boxplot('Item_Visibility',data=test)

sns.scatterplot(x= 'Item_Visibility', y='Item_Outlet_Sales',data=train)

# Item_Visibility has -0.12 correlation with Item_Outlet_Sales. 

#Item_Type

plt.figure(figsize=(25,7))
sns.countplot('Item_Type',data=train,palette='spring')

plt.figure(figsize=(10,8))
sns.barplot(y='Item_Type',x='Item_Outlet_Sales',data=train,palette='flag')

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Item_Type", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.0003. Null hypothesis of both these varaibles are independent to each other can be rejected.
import scipy.stats as stats

stats.shapiro(model.resid)

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Item_MRP has 0.567 correlation with Item_Outlet_Sales. 

#Outlet_Identifier

train.Outlet_Identifier.value_counts()

plt.figure(figsize=(8,5))
sns.countplot('Outlet_Identifier',data=train,palette='summer')
k = pd.crosstab(aggfunc="sum", index=train['Outlet_Identifier'], columns=train['Outlet_Identifier'], values=train['Item_Outlet_Sales'])

k_clms = k.columns

data = {}
for i in range(0, k.shape[0]):
    data[k_clms[i]] = k[k_clms[i]][i]

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Outlet_Identifier", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.0 Null hypothesis of both these varaibles are independent to each other can be rejected.
import scipy.stats as stats

stats.shapiro(model.resid)

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Outlet_Establishment_Year

#Adding new column with Tenure of the outlet upto 2022.
freq = []
for i in range(0,train.shape[0]):
   freq.append(2022 - train.Outlet_Establishment_Year[i])
train['Outlet_Tenure'] = freq

#Adding new column with Tenure of the outlet upto 2022.
freq_test = []
for j in range(0,test.shape[0]):
   freq_test.append(2022 - test.Outlet_Establishment_Year[j])

test['Outlet_Tenure'] = freq_test

sns.scatterplot(x= 'Outlet_Tenure', y='Item_Outlet_Sales',data=train)
# 0.05 correlation.

#Outlet_Size
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Size',data=train,palette='summer')

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Outlet_Size", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.0. Null hypothesis of both these varaibles are independent to each other can be rejected.
import scipy.stats as stats

stats.shapiro(model.resid)

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Outlet_Location_Type
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Location_Type',data=train,palette='autumn')

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Outlet_Location_Type", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.0. Null hypothesis of both these varaibles are independent to each other can be rejected.
import scipy.stats as stats
stats.shapiro(model.resid)

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Outlet_Type
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Type',data=train,palette='twilight')

plt.figure(figsize=(10,8))
sns.barplot(y='Item_Type',x='Item_Outlet_Sales',data=train,palette='flag')

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols("Item_Outlet_Sales ~ Outlet_Type", data= train).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

k = anova_table(aov_table)
# p value = 0.0. Null hypothesis of both these varaibles are independent to each other can be rejected.
import scipy.stats as stats
stats.shapiro(model.resid)

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

#Step 5: Building Model

# keep only Item_MRP and build the model.  RMSE = 1405.1238194114583.
# Keep Item_MRP & Outlet_Location_Type. RMSE = 1395.28
# Keep Item_MRP, Outlet_Location_Type & Outlet_Size. RMSE = 1393.56
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier & Outlet_Size. RMSE = 1273.42
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Type & Outlet_Size. RMSE = 1273.64. Item_Type is not have much significance
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Type, Item_Weight & Outlet_Size. RMSE = 1273.8. Item_Weight is not usefull
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Fat_Content & Outlet_Size. RMSE = 1272.91
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Fat_Content, Outlet_Type & Outlet_Size. RMSE = 1191.96
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Fat_Content, Outlet_Type, Item_Visibility & Outlet_Size. RMSE = 1188.38
# Keep Item_MRP, Outlet_Location_Type, Outlet_Identifier,Item_Fat_Content, Outlet_Type, Item_Visibility, Item_Identifier & Outlet_Size. RMSE = 1188.386740120419
''' Keep Item_MRP, Outlet_Location_Type, Item_Fat_Content, Outlet_Type, 'Outlet_Identifier,'Outlet_Size', ,'Item_Weight'. RMSE = '''

# Dropping insignificant features. Item_Weight, Outlet_Establishment_Year
train_new = train.drop(columns=[ 'Item_Type' ,'Item_Identifier','Outlet_Establishment_Year', 'Item_Visibility'])
test_new = test.drop(columns=['Item_Type','Item_Identifier' ,'Outlet_Establishment_Year','Item_Visibility'])

from scipy.stats import chi2_contingency
#Chi_square_test to find categorical vars importance.
stat, p, dof, expected = chi2_contingency(pd.crosstab(train.Outlet_Type,train.Outlet_Size, margins=True))

sns.boxplot('Item_Visibility',data = train)
k = pd.crosstab(aggfunc="sum", index=train['Item_Identifier'], columns=train['Item_Identifier'], values=train['Item_Outlet_Sales'])

# Labelencoding
le = LabelEncoder()
var_mod = train_new.select_dtypes(include='object').columns
for i in var_mod:
    train_new[i] = le.fit_transform(train_new[i])
    
for i in var_mod:
    test_new[i] = le.fit_transform(test_new[i])

# Seperate Features and Target
X= train_new.drop(columns = ['Item_Outlet_Sales'], axis=1)
y= train_new['Item_Outlet_Sales']
features= X.columns

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# 30% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=20)

# Model Building
LR = LinearRegression(normalize=True)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1188.46

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)

# round 2

# Dropping insignificant features. Item_Weight, Outlet_Establishment_Year
train_new = train.drop(columns=['Item_Weight','Outlet_Establishment_Year','Item_Identifier','Outlet_Tenure'])
test_new = test.drop(columns=['Item_Weight','Outlet_Establishment_Year', 'Item_Identifier','Outlet_Tenure'])

# Labelencoding
le = LabelEncoder()
var_mod = train_new.select_dtypes(include='object').columns
for i in var_mod:
    train_new[i] = le.fit_transform(train_new[i])
    
for i in var_mod:
    test_new[i] = le.fit_transform(test_new[i])

# Seperate Features and Target
X= train_new.drop(columns = ['Item_Outlet_Sales'], axis=1)
features= X.columns
y= train_new['Item_Outlet_Sales']

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)

# 20% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=20)

# Model Building
LR = LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1168.43

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)

# round 3

# Dropping insignificant features. Item_Weight, Outlet_Establishment_Year
train_new = train.drop(columns=['Item_Weight','Outlet_Establishment_Year','Item_Identifier','Outlet_Tenure','Item_Type'])
test_new = test.drop(columns=['Item_Weight','Outlet_Establishment_Year', 'Item_Identifier','Outlet_Tenure','Item_Type'])

# Labelencoding
le = LabelEncoder()
var_mod = train_new.select_dtypes(include='object').columns
for i in var_mod:
    train_new[i] = le.fit_transform(train_new[i])
    
for i in var_mod:
    test_new[i] = le.fit_transform(test_new[i])

# Seperate Features and Target
X= train_new.drop(columns = ['Item_Outlet_Sales'], axis=1)
features= X.columns
y= train_new['Item_Outlet_Sales']

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)

# 20% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=20)

# Model Building
LR = LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1168.43

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)

# Ridge 
from sklearn.linear_model import Ridge
regressor = Ridge(alpha=1.0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_valid)
coef = pd.Series(regressor.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(regressor.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1168.43

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)

#Lasso
from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=0.01)
Lasso.fit(X_train,y_train)

y_pred = Lasso.predict(X_valid)
coef = pd.Series(Lasso.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(Lasso.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1168.43

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)


# Round 4 with PCA

# Labelencoding
le = LabelEncoder()
var_mod = train.select_dtypes(include='object').columns
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
for i in var_mod:
    test[i] = le.fit_transform(test[i])

X= train.drop(columns = ['Item_Outlet_Sales'], axis=1)
y= train['Item_Outlet_Sales']

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)

# 20% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=20)

from sklearn.decomposition import PCA
pca = PCA(0.9)

X_train_Pca = pca.fit_transform(X_train) 
X_test_Pca = pca.transform(X_valid) 

pca.fit(X_train)
print(pca.components_)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()
 
explained_variance = pca.explained_variance_ratio_ 

#Last Round
cols = ['Item_Weight','Outlet_Establishment_Year','Item_Identifier','Item_Fat_Content', 'Item_Visibility']
train_new = train.drop(columns=cols)
test_new = test.drop(columns=cols)

# Labelencoding
le = LabelEncoder()
var_mod = train_new.select_dtypes(include='object').columns
for i in var_mod:
    train_new[i] = le.fit_transform(train_new[i])
    
for i in var_mod:
    test_new[i] = le.fit_transform(test_new[i])


# Seperate Features and Target
X= train_new.drop(columns = ['Item_Outlet_Sales'], axis=1)
features= X.columns
y= train_new['Item_Outlet_Sales']

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# 30% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=20)

# Model Building
LR = LinearRegression(normalize=True)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)

#RMSE
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse) # 1188.46

from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)
