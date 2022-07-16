import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

df_org = pd.read_csv('train.csv')

df = df_org
cols = df.columns

cat_var = []

for i in df_org.columns:
    if df_org[i].dtype=='O':
        cat_var.append(i)

# The cat variables are Ordinal data. So, it would be good to label encode them
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in cat_var:
    df[i] = encoder.fit_transform(df[i])
    

na = (df.isna().sum()/len(df))*100

sorted_NA = na.sort_values(ascending = False)

null_var_names = []
for i in range(0, len(sorted_NA)):
    if sorted_NA[i]>50.0:
        null_var_names.append(sorted_NA.index[i])

#Drop the columns with more than 50% missing values
df_new = df.drop(columns = null_var_names)

col_new = df_new.columns

#Replace the missing values with mean value
for i in range(0, df_new.shape[1]):
    m =df_new[col_new[i]].mean()
    df_new[col_new[i]].fillna(m, inplace=True)

l = df_new.isna().sum()
'''
from pycaret.regression import *
exp_reg101 = setup(data = df_new, target = 'rating_num', session_id=123, imputation_type='iterative',fold_shuffle=True) '''

X = df_new.drop(columns = ['rating_num', 'row_id'])
y = df_new.rating_num

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Standardize the data
'''from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)'''



# Evaluating other models
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

algos = [LinearRegression(),  Ridge(), Lasso(),
          KNeighborsRegressor(), DecisionTreeRegressor(),XGBRegressor(), LGBMRegressor() ]

names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
         'K Neighbors Regressor', 'Decision Tree Regressor', 'XG Boost', 'lightGBM']

r2_score_list_y1 = []

for name in algos:
    model = name
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_score_list_y1.append(r2)
   
evaluation_y1 = pd.DataFrame({'Model': names,
                           'RMSE': r2_score_list_y1})
# lightGBM is working  better.

#Lets test it on test data.

