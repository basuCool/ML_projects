import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

# import XGBoost classifier
from xgboost import XGBClassifier


df = pd.read_csv('uci-secom.csv')

#Remvoe the Time feature
df.drop(columns=['Time'],inplace=True)

y = df['Pass/Fail']

#Remove the target variable
df_new = df.drop('Pass/Fail', axis=1)


#Replace all nan with zero values
for col in df_new.columns.values:
    df_new[col].fillna(df_new[col].mean(),inplace=True)
    
#Remove constant columns
for col in df_new.columns.values:
    if df_new[col].var() == 0.0:
        del df_new[col]
        
df_new['Pass'] = y

df_new['Pass'].replace(to_replace= -1, value = 0, inplace=True)        

X = df_new.drop('Pass', axis=1)
y = df_new['Pass']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

 # Fitting XGBoost to training set
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Try Logistic regression to see the accuracy 
# Scale the data
#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced', max_iter=10000) 

#log = LogisticRegression(max_iter=10000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
#Creating Confusion matrix
cm_log= confusion_matrix(y_test, y_pred)

#Target variable is highly imbalanced. Balance it with SMOTE

from imblearn.over_sampling import SMOTE

X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)

X_test_smote, y_test_smote = SMOTE().fit_resample(X_test, y_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression( max_iter=10000) 

#log = LogisticRegression(max_iter=10000)

logreg.fit(X_train_smote, y_train_smote)

y_pred_smote = logreg.predict(X_test_smote)
#Creating Confusion matrix
cm_log_smote= confusion_matrix(y_test_smote, y_pred_smote)


from lightgbm import LGBMClassifier

lgbcls = LGBMClassifier()

lgbcls.fit(X_train_smote, y_train_smote)

y_pred = lgbcls.predict(X_test_smote)
#Creating Confusion matrix
cm_lgbm= confusion_matrix(y_test_smote, y_pred)


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()

random.fit(X_train_smote, y_train_smote)

y_pred = random.predict(X_test_smote)
#Creating Confusion matrix
cm_Random= confusion_matrix(y_test_smote, y_pred)

 # Fitting XGBoost to training set
clf = XGBClassifier()
clf.fit(X_train_smote, y_train_smote)

y_pred = clf.predict(X_test_smote)
#Creating Confusion matrix
cm_xgb= confusion_matrix(y_test_smote, y_pred)

