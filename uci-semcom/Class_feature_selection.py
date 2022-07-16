# C:\Users\teggiba\OneDrive - Lam Research\Desktop\ML\Exercises\uci-semcom

# Feature selection dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import ADASYN, SMOTE 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


df = pd.read_csv('uci-secom.csv')

#drop date column
df = df.drop(columns=['Time'])

# Remove constant value columns
df_new = df.select_dtypes(exclude=['object'])
df_new = df_new.loc[:, df_new.var() != 0.0]

df_new['Pass/Fail'].replace(to_replace= -1, value = 0, inplace=True)        

# Seperating features and result
y = df_new['Pass/Fail']
X = df_new.drop(columns = ['Pass/Fail'])

# Checking the result distribuition. 
k = y.value_counts()
# NOTE: y is imbalanced. i.e. 1463(1) vs 104 (-1). This will not yeild good model.

#Fill NAN values with mean of particular column values
X = X.fillna(X.mean())


sm = SMOTE(random_state=0)
# f1 = 0.925 auc = 0.842 with logisticReg.
# f1 = 0.996 auc = 1.0 with SVM
# f1 = 0.991 auc = 0.999 with SVM (with PCA)
ad = ADASYN(random_state=0) 
# f1 = 0.93 auc = 0.853 with logisticReg
# f1 = 0.995 auc = 1.0 with SVM
# f1 = 0.998 auc = 1.000 with SVM (with PCA)
X_new, y_new = sm.fit_sample(X, y)
X_new2, y_new2 = ad.fit_sample(X, y)


steps = [('scaler', StandardScaler()),('pca', PCA(0.90)),('clf', SVC())]

# Applying PCA
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X_new)
X_std2 = stdsc.fit_transform(X_new2)

"""pca = PCA()
pca.fit(X_std)
pca_data = pca.transform(X_std)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)

labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1, len(per_var)+1),height=per_var, tick_label = labels)
plt.ylabel('% of Explained Variance')
plt.xlabel('PC')
plt.title('Scree Plot')
plt.show()"""

pca = PCA(0.90)
pca.fit(X_std2)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

# 121 components explain 90% of variance for SMOTE & ADASYN sampling
X_new_pca_90_sm = pca.transform(X_std)
X_new_pca_90_ad = pca.transform(X_std2)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new_pca_90_ad, y_new2, test_size = 0.2, random_state=0)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""
# import logistic regression model and accuracy_score metric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = LogisticRegression(solver = 'lbfgs')

# fit the model
clf.fit(X_train, Y_train)
# prediction for training dataset
train_pred = clf.predict(X_train)
# prediction for testing dataset
test_pred = clf.predict(X_test)
"""
 # Fitting SVM to traing set

clf = SVC(kernel = 'rbf', random_state = 0, probability=True, gamma='auto')
# fit the model
clf.fit(X_train, Y_train)

# prediction for training dataset
train_pred = clf.predict(X_train)
# prediction for testing dataset
test_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, test_pred)

print('Accuracy score for Training Dataset = ', accuracy_score(train_pred, Y_train))
print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, Y_test))

# predict probabilities
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# Predicting the Test set results
y_pred = clf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(Y_test, lr_probs)
lr_f1, lr_auc = f1_score(Y_test, y_pred), auc(lr_recall, lr_precision)
# summarize scores
print('SVM: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='SVC')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


