# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:13:13 2020

@author: teggiba
"""
print("ML")

"""
Acute inflammation data classification code
"""
#C:\builds\ML\Exercises\Acute_inflammation_data

import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', sep=',',header=None)
df.columns = ["temp", "nausea", "LumbarPain","Urine",
              "MicturitionPain","BurningPain","Y1","Y2" ]
#Y1 =  decision: Inflammation of urinary bladder
#Y2 = decision: Nephritis of renal pelvis origin 

x = df.iloc[:,0:6]
y1 = df.iloc[:,6]
y2 = df.iloc[:,-1]


index = np.arange(len(df))

#EDA
import seaborn as sns

sns.barplot(x='temp', y= 'Y1', data=df, hue= 'nausea')
sns.barplot(x='temp', y= 'Y2', data=df, hue= 'nausea')

sns.barplot(x='temp', y= 'Y1', data=df, hue= 'LumbarPain')
sns.barplot(x='temp', y= 'Y2', data=df, hue= 'LumbarPain')

sns.barplot(x='temp', y= 'Y1', data=df, hue= 'Urine')
sns.barplot(x='temp', y= 'Y2', data=df, hue= 'Urine')

sns.barplot(x='temp', y= 'Y1', data=df, hue= 'MicturitionPain')
sns.barplot(x='temp', y= 'Y2', data=df, hue= 'MicturitionPain')

sns.barplot(x='temp', y= 'Y1', data=df, hue= 'BurningPain')
sns.barplot(x='temp', y= 'Y2', data=df, hue= 'BurningPain')

# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()

X1 = x
for i in range(1, 6):
    X1.iloc[:,i] = labelEncoder_X.fit_transform(X1.iloc[:,i])
y1 = labelEncoder_y.fit_transform(y2)

# Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVM to traing set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Creating Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


