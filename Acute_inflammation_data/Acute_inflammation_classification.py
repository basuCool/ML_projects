# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
from statsmodels.stats import weightstats as stests

df = pd.read_csv('data.csv', sep=',',header=None)
df.columns = ["temp", "nausea", "LumbarPain","Urine","MicturitionPain","BurningPain","Y1","Y2" ]
# Inflammation of urinary bladder
y1 = df.iloc[:,6]
# Nephritis of renal pelvis origin
y2 = df.iloc[:,7]
X = df.iloc[:,0:6]

index = np.arange(len(df))
#X[0] = Temperature of patient { 35C-42C }
sns.distplot(df['temp'], bins=7)
sns.boxplot(x="temp",y="Y1",data=df)
sns.catplot(x="temp", hue="Y1", kind="count", data=df);
sns.catplot(x="temp", hue="Y2", kind="count", data=df);
# Occurence of nausea
sns.barplot(x=index,y="nausea",data=df)
sns.countplot(x="nausea", hue="Y1",data=df)
sns.countplot(x="nausea", hue="Y2",data=df)
#Lumbar pain
sns.barplot(x=index,y="LumbarPain",data=df)
sns.countplot(x="LumbarPain", hue="Y1",data=df)
sns.countplot(x="LumbarPain", hue="Y2",data=df)
#Urine pushing
sns.barplot(x=index,y="Urine",data=df)
sns.countplot(x="Urine", hue="Y1",data=df)
sns.countplot(x="Urine", hue="Y2",data=df)
#Micturition Pains
sns.barplot(x=index,y="MicturitionPain",data=df)
sns.countplot(x="MicturitionPain", hue="Y1",data=df)
sns.countplot(x="MicturitionPain", hue="Y2",data=df)
#Burning sensation
sns.barplot(x=index,y="BurningPain",data=df)
sns.countplot(x="BurningPain", hue="Y1",data=df)
sns.countplot(x="BurningPain", hue="Y2",data=df)

#Creating categorical variable from df
cat_var = df.select_dtypes(include=['object']).copy()
cols = list(cat_var.columns)
p_value= []
p2_value= []
#Chi_square_test to find cat vars importance

pd.crosstab(cat_var.nausea, df.Y1, margins=True)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.nausea, df.Y1, margins=True))
p_value.append(p) 
pd.crosstab(cat_var.nausea, df.Y2, margins=True)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.nausea, df.Y2, margins=True))
p2_value.append(p)

stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.LumbarPain, df.Y1, margins=True))
p_value.append(p)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.LumbarPain, df.Y2, margins=True))
p2_value.append(p)

stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.Urine, df.Y1, margins=True))
p_value.append(p)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.Urine, df.Y2, margins=True))
p2_value.append(p)

stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.MicturitionPain, df.Y1, margins=True))
p_value.append(p)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.MicturitionPain, df.Y2, margins=True))
p2_value.append(p)

stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.BurningPain, df.Y1, margins=True))
p_value.append(p)
stat, p, dof, expected = chi2_contingency(pd.crosstab(cat_var.BurningPain, df.Y2, margins=True))
p2_value.append(p)

# Z-test for numerical-categorical
grouped_df = df.groupby('Y1')
gb = grouped_df.groups
ztest ,pval = stests.ztest(df.temp[gb['no']], x2 = df.temp[gb['yes']], value=0,alternative='two-sided')
print(float(pval))
if pval < 0.005 :
    print('temp variable is Dependent for Y1')
else:
    print('temp variable is Independent for Y1')
# Z-test for numerical-categorical
grouped_df = df.groupby('Y2')
gb = grouped_df.groups
ztest ,pval = stests.ztest(df.temp[gb['no']], x2 = df.temp[gb['yes']], value=0,alternative='two-sided')
print(float(pval))
if pval < 0.005 :
    print('temp variable is Dependent for Y2')
else:
    print('temp variable is Independent for Y2')
    
for i in range(5):
    if(p_value[i]<=0.05):
        print(cols[i],"is Dependent for Y1")
    else:
        print(cols[i], "is Independent for Y1")
for i in range(5):
    if(p2_value[i]<=0.05):
        print(cols[i],"is Dependent for Y2")
    else:
        print(cols[i], "is Independent for Y2")
        
# Creating final X1, y1 dataframe with dependent variables
X1 = df.iloc[:,2:6]
# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()
cols2 = [0,1,2,3]
for i in cols2:
    X1.iloc[:,i] = labelEncoder_X.fit_transform(X1.iloc[:, i])
y1 = labelEncoder_y.fit_transform(y1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

# Fitting SVM to traing set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Creating Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Creating final X2, y2 dataframe with dependent variables
X2 = df.iloc[:,0:6].values
# Removing Urine and MicturitionPain feature variables from the X2
for i in range(2):
    X2 = np.delete(X2,3,1) #To delete redundent feature variables
# Encoding the categorical feature variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()
cols2 = [1,2,3]
for i in cols2:
    X2[:,i] = labelEncoder_X.fit_transform(X2[:, i])
y2 = labelEncoder_y.fit_transform(y2)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

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