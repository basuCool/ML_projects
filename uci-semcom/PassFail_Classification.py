import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv('uci-secom.csv')
df.fillna(df.mean(), inplace=True)

del df['Time']
#Remove constant columns
for col in df.columns.values:
    if df[col].var() == 0.0:
        del df[col]
df['Pass/Fail'].replace(to_replace= -1, value = 0, inplace=True)        
X = df.iloc[:,0:474].values
y = df.iloc[:, -1].values

def makeOverSamplesADASYN(X,y):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
 from imblearn.over_sampling import ADASYN 
 sm = ADASYN()
 X, y = sm.fit_sample(X, y)
 return(X,y)
X,y = makeOverSamplesADASYN(X,y)
 
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

#Applying PCA
pca = PCA(0.90)        
pca.fit(X_std)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
# n_components are 225
pca2 = PCA(n_components=121)
pca2.fit(X_std)
X_pca = pca.transform(X_std)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 0)

 # Fitting SVM to traing set
from sklearn.svm import SVC
clf2 = SVC(kernel = 'rbf', random_state = 0, probability=True, gamma='auto')
clf2.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# predict probabilities
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

lr_probs = clf2.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# Predicting the Test set results
y_pred = clf2.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
# summarize scores
print('SVM: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='SVC')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

pipe = Pipeline(steps=[('pca',pca),('clf',clf)])
params_grid = {
        'pca_n_components':[2,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,150,200],
        }



