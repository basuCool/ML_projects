import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats import weightstats as stests
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df = pd.read_csv('adult.csv', header=None)

df.columns = ["age", "workclass", "fnlwgt","education","education-num","marital-status",
              "occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week",
              "native-country", "y"]
df = df.replace(to_replace = ' >50K', 
                 value ='True') 
df = df.replace(to_replace = " <=50K", 
                 value = 'False')

df = df.replace(to_replace = " ?",value = np.nan)
df.isna().sum()


#Taking care of the missing data

cols = ['workclass', 'occupation', 'native-country']

for i in cols:  
    m = df[i].mode()
    df[i].fillna(m[0], inplace=True)

df.isna().sum()

df_new = df
#1 Age Numerical Variable Analysis
sns.countplot(df['age'])
sns.boxplot(x=df['age'])
sns.distplot(x=df['age'])

#Age is having right skewed distibution.

#Check for outliers
Q1 = df.age.quantile(0.25)
Q3 = df.age.quantile(0.75)

IQR = Q3-Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit

outliers_age = df[(df.age<lower_limit)|(df.age>upper_limit)]
# Cap the outliers with upper limit
df.age = df.age.apply(lambda x: upper_limit if x>upper_limit else x)

sns.countplot(df['age'])
sns.boxplot(x=df['age'])
sns.distplot(x=df['age'])

#Lets check for significance of age over target variable
grouped_df = df.groupby('y')
gb = grouped_df.groups
ztest ,pval = stests.ztest(df['age'][gb['False']], x2 = df['age'][gb['True']], value=0,alternative='two-sided')

#pval = 0.0. Thus we can say that age values mean value is not same for y class groups. i.e. There is siginificance which age is having over y classes.

#2 workclass categorical variable analysis
index = np.arange(len(df))

#workclass
sns.barplot(x=df['workclass'], y=index, data=df)
sns.countplot(x="workclass", hue="y", data=df)

stat, p, dof, expected = chi2_contingency(pd.crosstab(df.workclass, df.y, margins=True))

# P value = 2.504 e-186. i.e. Workclass is having significance over target variable.

#3 Final weight numerical  variable analysis
sns.boxplot(x=df['fnlwgt'])
sns.distplot(df['fnlwgt'], bins=5)

#Final weight is having right skewed distibution.
#Check for outliers
Q1 = df.fnlwgt.quantile(0.25)
Q3 = df.fnlwgt.quantile(0.75)

IQR = Q3-Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit

outliers_age = df[(df.fnlwgt<lower_limit)|(df.fnlwgt>upper_limit)]
# Cap the outliers with upper limit
df.fnlwgt = df.fnlwgt.apply(lambda x: upper_limit if x>upper_limit else x)

sns.boxplot(x=df['fnlwgt'])
sns.distplot(df['fnlwgt'], bins=5)

df.fnlwgt.describe()
#Lets check for significance of fnlwgt over target variable
grouped_df = df.groupby('y')
gb = grouped_df.groups
ztest ,pval = stests.ztest(df_new['fnlwgt'][gb['False']], x2 = df_new['fnlwgt'][gb['True']], value=0,alternative='two-sided')
# pval = 0.13017. fnlwgt is not significant over target variable.

#4 Marital-status categorical variable analysis
sns.barplot(x=df['marital-status'], y=index, data=df)
sns.countplot(x="marital-status", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['marital-status'], df.y, margins=True))
# p value = 0.0. Reject the null hypothesis and accept the Alternate hypothesis that marital status has significance over Target variable.

#5 education categorical variable analysis
sns.barplot(x=df['education'], y=index, data=df)
sns.countplot(x="education", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['education'], df.y, margins=True))
# p value = 0.0. Reject the null hypothesis and accept the Alternate hypothesis that education has significance over Target variable.

#5 occupation categorical variable analysis
sns.barplot(x=df['occupation'], y=index, data=df)
sns.countplot(x="occupation", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['occupation'], df.y, margins=True))
# p value = 0.0. Reject the null hypothesis and accept the Alternate hypothesis that occupation has significance over Target variable.

#6 relationship categorical variable analysis
sns.barplot(x=df['relationship'], y=index, data=df)
sns.countplot(x="relationship", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['relationship'], df.y, margins=True))
# p value = 0.0. Reject the null hypothesis and accept the Alternate hypothesis that relationship has significance over Target variable.

#7 race categorical variable analysis
sns.barplot(x=df['race'], y=index, data=df)
sns.countplot(x="race", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['race'], df.y, margins=True))
# p value = 4.40 e-65. Reject the null hypothesis and accept the Alternate hypothesis that race has significance over Target variable.

#8 sex categorical variable analysis
sns.barplot(x=df['sex'], y=index, data=df)
sns.countplot(x="sex", hue="y", data=df)
stat, p, dof, expected = chi2_contingency(pd.crosstab(df['sex'], df.y, margins=True))
# p value = 0.0. Reject the null hypothesis and accept the Alternate hypothesis that sex has significance over Target variable.

#9 capital-gain numerical variable analysis
sns.distplot(df['capital-gain'], bins=5)
sns.boxplot(x="capital-gain", data=df)

#Lets check for significance of capital-gain over target variable
grouped_df = df.groupby('y')
gb = grouped_df.groups
ztest ,pval = stests.ztest(df['capital-gain'][gb['False']], x2 = df['capital-gain'][gb['True']], value=0,alternative='two-sided')
# pval = 0.0. capital-gain is significant over target variable.

#10 capital-loss numerical variable analysis
sns.distplot(df['capital-loss'], bins=5)
sns.boxplot(x="capital-loss", hue="y", data=df)
ztest ,pval = stests.ztest(df['capital-loss'][gb['False']], x2 = df['capital-loss'][gb['True']], value=0,alternative='two-sided')
# pval = 3.573 e -166 capital-loss is significant over target variable.

#11 hours-per-week numerical variable analysis
sns.boxplot(x=df['hours-per-week'])
sns.distplot(df['hours-per-week'], bins=5)
df['hours-per-week'].describe()
# The distribuition is normal
#Check for outliers
Q1 = df['hours-per-week'].quantile(0.25)
Q3 = df['hours-per-week'].quantile(0.75)

IQR = Q3-Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit

outliers_age = df[(df['hours-per-week']<lower_limit)|(df['hours-per-week']>upper_limit)]

# Cap the outliers with upper limit
df['hours-per-week'] = df['hours-per-week'].apply(lambda x: upper_limit if x>upper_limit else (lower_limit if x<lower_limit else x) )
df['hours-per-week'].describe()

ztest ,pval = stests.ztest(df['hours-per-week'][gb['False']], x2 = df['hours-per-week'][gb['True']], value=0,alternative='two-sided')
# pval = 0.0. hours-per-week is significant over target variable

#12 native-country categorical variable analysis
sns.barplot(x=df['native-country'], y=index, data=df)
sns.countplot(df['native-country'])

stat, p, dof, expected = chi2_contingency(pd.crosstab(df['native-country'], df.y, margins=True))
# pval = 2.25 e-29. Country has significance over Target variable


X = df.iloc[:,0:14]
y = df.iloc[:,-1]

#Creating categorical variable from df
cat_var = X.select_dtypes(include=['object']).copy()
cols2 = list(cat_var.columns)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()

for i in cols2:
    X_train[i] = labelEncoder_X.fit_transform(X_train[i])
    X_test[i] = labelEncoder_X.transform(X_test[i])
    
y = labelEncoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Since the target variable is unbalanced. Need to oversample
# transform the dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

#######################################
# MODEL BUIDLING
#######################################

# Fitting SVM to traing set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

#Creating Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)

# Fitting XGBoost to training set
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

