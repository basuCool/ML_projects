# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/priyadarshi0007/ASTCapstone/master/AppointmentData.csv')

sns.heatmap(df.isnull())
df = df.dropna()

sns.boxplot(x=df['Age'])
index = np.arange(len(df))
sns.distplot(df['Age'], kde=True, bins=5)
sns.catplot(x="Age", hue="No-show", kind="count", data=df);

# Need to remove outliners
df = df[(df.Age>0)&(df.Age<=100)]
df.rename(columns = {'No-show':'Noshow'}, inplace = True)
import datetime
from datetime import datetime
#Changing the ScheduledDay/AppointmentDay to date and time
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['ScheduledDay_year'] = df['ScheduledDay'].dt.year
df['ScheduledDay_month'] = df['ScheduledDay'].dt.month
df['ScheduledDay_quarter'] = df['ScheduledDay'].dt.quarter
df['ScheduledDay_day'] = df['ScheduledDay'].dt.day # day number in month
df['ScheduledDay_hour'] = df['ScheduledDay'].dt.hour # this only give numer
df['ScheduledDay_dayOfTheWeek'] = df['ScheduledDay'].dt.weekday_name

df['AppointmentDay_year'] = df['AppointmentDay'].dt.year
df['AppointmentDay_month'] = df['AppointmentDay'].dt.month
df['AppointmentDay_quarter'] = df['AppointmentDay'].dt.quarter
df['AppointmentDay_day'] = df['AppointmentDay'].dt.day
df['AppointmentDay_hour'] = df['AppointmentDay'].dt.hour # this only give numer
df['AppointmentDay_dayOfTheWeek'] = df['AppointmentDay'].dt.weekday_name

df['difference'] = abs(df['AppointmentDay']- df['ScheduledDay'])
df['difference'] = df['difference'].apply(lambda x: x.total_seconds()/(3600*24))

sns.distplot(df['Age'], kde=False);

h0 = df[df['Hipertension']==0.].groupby(['Noshow'])[['Hipertension']].count()
h1 = df[df['Hipertension']==1.].groupby(['Noshow'])[['Hipertension']].count()

d0 = df[df['Diabetes']==0.].groupby(['Noshow'])[['Diabetes']].count()
d1 = df[df['Diabetes']==1.].groupby(['Noshow'])[['Diabetes']].count()

a0 = df[df['Alcoholism']==0.].groupby(['Noshow'])[['Alcoholism']].count()
a1 = df[df['Alcoholism']==1.].groupby(['Noshow'])[['Alcoholism']].count()
# concatenating the columns
resulthda0 = pd.concat([h0, d0, a0], axis=1, sort=False)

sns.countplot(x="Noshow", order = df['Noshow'].value_counts().index,hue="ScheduledDay_dayOfTheWeek", data=df)

df['ScheduledDay_day'] = df['ScheduledDay'].dt.weekday # day number in week
df['AppointmentDay_day'] = df['AppointmentDay'].dt.weekday

#We need to handle the Object data type to fit for the model
df2 = df.drop(columns=['PatientId',
                                    'ScheduledDay',
                                    'AppointmentDay',
                                    'AppointmentID',
                                   ])
X = df2.drop(columns=['Noshow'])      
y = df2.iloc[:,9]

cat_var = df2.select_dtypes(include=['object']).copy()
p_value= []
#Chi_square_test to find cat vars importance
from scipy.stats import chi2_contingency
from statsmodels.stats import weightstats as stests
#Gender vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.Gender, cat_var.Noshow, margins=True))
p_value.append(p)
# Neighbourhood vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.Neighbourhood, cat_var.Noshow, margins=True))
p_value.append(p)
#  ScheduledDay_dayOfTheWeek vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var['ScheduledDay_dayOfTheWeek'], cat_var.Noshow, margins=True))
p_value.append(p)
#  AppointmentDay_dayOfTheWeek vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var['AppointmentDay_dayOfTheWeek'], cat_var.Noshow, margins=True))
p_value.append(p)
#  Diabetes vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['Diabetes'], df.Noshow, margins=True))
p_value.append(p)
#  Handcap vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['Handcap'], df.Noshow, margins=True))
p_value.append(p)
#  SMS_received vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['SMS_received'], df.Noshow, margins=True))
p_value.append(p)
#  Scholarship vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['Scholarship'], df.Noshow, margins=True))
p_value.append(p)
#  Hipertension vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['Hipertension'], df.Noshow, margins=True))
p_value.append(p)
#  Alcoholism vs y
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['Alcoholism'], df.Noshow, margins=True))
p_value.append(p)

# Z-test for numerical-categorical
grouped_df = df2.groupby('Noshow')
gb = grouped_df.groups
int_var = df2.select_dtypes(include=['int64']).copy()
float_var = df2.select_dtypes(include=['float64']).copy()
p_value_2 = []
#Diabetes 
ztest ,pval = stests.ztest(df2['Diabetes'][gb['No']], x2 = df2['Diabetes'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#Handcap
ztest ,pval = stests.ztest(df2['Handcap'][gb['No']], x2 = df2['Handcap'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#SMS_received
ztest ,pval = stests.ztest(df2['SMS_received'][gb['No']], x2 = df2['SMS_received'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#ScheduledDay_year
ztest ,pval = stests.ztest(df2['ScheduledDay_year'][gb['No']], x2 = df2['ScheduledDay_year'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#ScheduledDay_month
ztest ,pval = stests.ztest(df2['ScheduledDay_month'][gb['No']], x2 = df2['ScheduledDay_month'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#ScheduledDay_quarter
ztest ,pval = stests.ztest(df2['ScheduledDay_quarter'][gb['No']], x2 = df2['ScheduledDay_quarter'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#ScheduledDay_day
ztest ,pval = stests.ztest(df2['ScheduledDay_day'][gb['No']], x2 = df2['ScheduledDay_day'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#ScheduledDay_hour
ztest ,pval = stests.ztest(df2['ScheduledDay_hour'][gb['No']], x2 = df2['ScheduledDay_hour'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#AppointmentDay_year only have one value.. need to exclude
#ztest ,pval = stests.ztest(df2['AppointmentDay_year'][gb['No']], x2 = df2['AppointmentDay_year'][gb['Yes']], value=0,alternative='two-sided')
#p_value_2.append(pval)
#AppointmentDay_month
ztest ,pval = stests.ztest(df2['AppointmentDay_month'][gb['No']], x2 = df2['AppointmentDay_month'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#AppointmentDay_quarter... only have one value.. need to exclude
#ztest ,pval = stests.ztest(df2['AppointmentDay_quarter'][gb['No']], x2 = df2['AppointmentDay_quarter'][gb['Yes']], value=0,alternative='two-sided')
#p_value_2.append(pval)
#AppointmentDay_day
ztest ,pval = stests.ztest(df2['AppointmentDay_day'][gb['No']], x2 = df2['AppointmentDay_day'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#AppointmentDay_hour.. only have one value.. need to exclude
#ztest ,pval = stests.ztest(df2['AppointmentDay_hour'][gb['No']], x2 = df2['AppointmentDay_hour'][gb['Yes']], value=0,alternative='two-sided')
#p_value_2.append(pval)
#Age
ztest ,pval = stests.ztest(df2['Age'][gb['No']], x2 = df2['Age'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#Scholarship
ztest ,pval = stests.ztest(df2['Scholarship'][gb['No']], x2 = df2['Scholarship'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#Hipertension
ztest ,pval = stests.ztest(df2['Hipertension'][gb['No']], x2 = df2['Hipertension'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#Alcoholism having >0.05 p-value
ztest ,pval = stests.ztest(df2['Alcoholism'][gb['No']], x2 = df2['Alcoholism'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)
#difference
ztest ,pval = stests.ztest(df2['difference'][gb['No']], x2 = df2['difference'][gb['Yes']], value=0,alternative='two-sided')
p_value_2.append(pval)

# Exculde Gender,AppointmentDay_dayOfTheWeek, AppointmentDay_year, AppointmentDay_quarter, AppointmentDay_hour and Alcoholism from X
X2 = X.drop(columns=['Gender',
                     'ScheduledDay_dayOfTheWeek',
                     'Handcap',
                     'AppointmentDay_year',
                     'AppointmentDay_quarter',
                     'AppointmentDay_hour',
                     'Alcoholism',
                                   ])
# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()
X2.iloc[:,1] = labelEncoder_X.fit_transform(X2.iloc[:, 1])
X2.iloc[:,13] = labelEncoder_X.fit_transform(X2.iloc[:, 13])
y = labelEncoder_y.fit_transform(y)
#One hot encoding for categorical features
onehotencoder = OneHotEncoder(categorical_features = [13])
X2 = onehotencoder.fit_transform(X2).toarray()
# Avoding the dummy variable trap by ignoring the one column from onehotencoded X
X2 = X2[:, 1:]
#One hot encoding for categorical features Neighbourhood
onehotencoder = OneHotEncoder(categorical_features = [6])
X2 = onehotencoder.fit_transform(X2).toarray()
# Avoding the dummy variable trap by ignoring the one column from onehotencoded X
X2 = X2[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=10, n_jobs=-1,max_depth=10)
# Train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# View The Accuracy Of Our Full Feature (4 Features) Model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.decomposition import PCA 
pca = PCA(n_components = 6) 
  
X_train_Pca = pca.fit_transform(X_train) 
X_test_Pca = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_ 

clf_important = RandomForestClassifier(n_estimators=10000, random_state=10, n_jobs=-1,max_depth=10)

clf_important.fit(X_train_Pca, y_train)
y_pred = clf_important.predict(X_test_Pca)
accuracy_score(y_test, y_pred)

import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(X_train, y_train)
model.score(X_test,y_test)

for feature in zip(X, clf.feature_importances_):
    print(feature)
    
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(clf, threshold=0.15)
# Train the selector
sfm.fit(X_train, y_train)
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=10, n_jobs=-1,max_depth=10)
# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

y_important_pred = clf_important.predict(X_important_test)

accuracy_score(y_test, y_important_pred)
# Fitting SVM to traing set
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf', random_state = 0, probability=True)
classifier2.fit(X_train, y_train)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from matplotlib import pyplot

# predict probabilities
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# Predicting the Test set results
y_pred = clf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='RF')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()