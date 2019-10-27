# -*- coding: utf-8 -*-
#1 Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_auc_score, roc_curve, precision_recall_curve, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.externals import joblib
from xgboost import XGBClassifier, plot_importance, plot_tree

#ignore warnings during runing models
import warnings
warnings.filterwarnings('ignore')

#2 Load Dataset
#load whole dataset
MONCs = pd.read_excel('MONCs Exp SH20190721.csv').drop(['No'], axis =1)

'''
#3 Summarize the Dataset
#3.1 Dimensions of Dataset
print('============================================================')
print("The shape of MONCs is:\n")
print(MONCs.shape)

#3.2 Peek at the Data
print('============================================================')
print('The first five rows of MONCs are:\n')
print(MONCs.head())

#3.3 Statistical Summary
print('============================================================')
print("The description of MONCs is:\n")
print(MONCs.describe())

#3.4 Check Info 
print('============================================================')
print("The brief information of MONCs is:\n")
print(MONCs.info())

#3.5 Check Missing Information
print('============================================================')
print("Number of MONCs is null:\n")
print(MONCs.isnull().sum())

#3.6 Class Distribution
print('============================================================')
print("Class distribution of MONCs is:\n")
print(MONCs['P'].value_counts())
print('============================================================')

#4 Data Visualization
#4.1 Univariate Plots
plt.figure(figsize = (20, 8))
MONCs.plot(kind='box', subplots=True, layout=(4,7), sharex=False, sharey=False)
plt.show()

#4.2 Histogram of each input variable
plt.figure(figsize = (10, 8))
MONCs.hist()
plt.show()

#4.3 Multivariate Plots
#scatter plot matrix
scatter_matrix(MONCs)
plt.show()

#4.4 Draw the counts of SC and NSC
distribution = sns.countplot(x = 'P', data = MONCs)
plt.show()

#Correlation Analysis
corrMatt = MONCs[["PgCx",'Rchainlength', 'R_OH', "C_Atomicweight","C_Charge","C_Ionicradius", "A_Atomicweight", "A_Charge", "A_Ionicradius", "DMF", 
                 "DMF_Polarity", "Methanol", "Methanol_Polarity", "Acetonitrile", "Acetonitrile_Polarity", "H2O", "H2O_Polarity", 
                 "Modulator_Molar_Mass", "Modulator_pKa", "Modulator_Amount", "Reaction_Temperature", "Product"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,20)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()
'''

#5 Evaluate Some Algorithms

#5.1 Split whole dataset into traning and test dataset
#Features and target
X = MONCs.drop(['P'], axis = 1)
y = MONCs['P'].astype('category')

#split dataset into training and test set
random_state = 21
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = random_state)

#Normalization 
mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

#load validational data 
MONCs_valexp = pd.read_excel('MONCs ValExp SH20190721.xlsx').drop(['No'], axis =1)

#define features and target
X_valexp = MONCs_valexp.drop(['P'], axis = 1)
y_valexp = MONCs_valexp['P']

#Normarlization
X_valexp_scaled = mms.transform(X_valexp)

#define classification report
def clfr(y_test, y_pred):
    print('(1) The predictive accuracy is: ', accuracy_score(y_test, y_pred))
    print('(2) Classification report is:\n', classification_report(y_test, y_pred))
    print('(3) Confusion matrix is:\n', confusion_matrix(y_test, y_pred))  


cv = StratifiedKFold(n_splits = 5, random_state = random_state)
scoring = 'accuracy'

#M1 Logistic Regression Classifier
lr = LogisticRegression()
param_grid = {'penalty': ['l2'],
              'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
              'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(lr, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('1 Logistic Regression Classifier')
print('================================')
print('1.1 GridSearchCV results:')
print('1.1.1 Best parameters:')
print(grid.best_params_)
print('1.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('1.2 Test Dataset')
y_pred_lr = grid.predict(X_test)
clfr(y_test, y_pred_lr)
print('================================')
print('1.3 Training Dataset')
y_pred_lr_train = grid.predict(X_train)
clfr(y_train, y_pred_lr_train)

#M2 Gaussian Naive Bayes Classifier
gnb = GaussianNB()
param_grid = {}
grid = GridSearchCV(gnb, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('2 Gaussian Naive Bayes Classifier')
print('================================')
print('2.1 GridSearchCV results:')
print('2.1.1 Best parameters:')
print(grid.best_params_)
print('2.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('2.2 Test Dataset')
y_pred_gnb = grid.predict(X_test)
clfr(y_test, y_pred_gnb)
print('================================')
print('2.3 Training Dataset')
y_pred_gnb_train = grid.predict(X_train)
clfr(y_train, y_pred_gnb_train)

#M3 k-Nearest Neighbors Classifier
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 11),
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree']}
grid = GridSearchCV(knn, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('3 k-Nearest Neighbors Classifier')
print('================================')
print('3.1 GridSearchCV results:')
print('3.1.1 Best parameters:')
print(grid.best_params_)
print('3.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('3.2 Test Dataset')
y_pred_knn = grid.predict(X_test)
clfr(y_test, y_pred_knn)
print('================================')
print('3.3 Training Dataset')
y_pred_knn_train = grid.predict(X_train)
clfr(y_train, y_pred_knn_train)

#M4 Support Vector Machine Classifier
svm = SVC()
param_grid = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
              'gamma': ['auto']},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(svm, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('4 Support Vector Machine Classifier')
print('================================')
print('4.1 GridSearchCV results:')
print('4.1.1 Best parameters:')
print(grid.best_params_)
print('4.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('4.2 Test Dataset')
y_pred_svm = grid.predict(X_test)
clfr(y_test, y_pred_svm)
print('================================')
print('4.3 Training Dataset')
y_pred_svm_train = grid.predict(X_train)
clfr(y_train, y_pred_svm_train)

#M5 Decision Tree Classifier
dt = DecisionTreeClassifier(random_state = random_state)
param_grid = {'criterion':['gini', 'entropy'], 
              'max_depth': range(3, 11),
              }
grid = GridSearchCV(dt, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('5 Decision Tree Classifier')
print('================================')
print('5.1 GridSearchCV results:')
print('5.1.1 Best parameters:')
print(grid.best_params_)
print('5.1.2 Best Estimator:')
print(grid.best_estimator_)
y_pred_dt = grid.predict(X_test)
print('================================')
print('5.2 Test Dataset')
clfr(y_test, y_pred_dt)
y_pred_dt_train = grid.predict(X_train)
print('================================')
print('5.3 Training Dataset')
clfr(y_train, y_pred_dt_train)

#M6 Random Forest Classifier
rf = RandomForestClassifier(random_state = 21)
param_grid = {'max_depth': range(3, 8),
              'n_estimators': [10, 100, 200, 500, 1000]
              }
grid = GridSearchCV(rf, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('6 Random Forest Classifier')
print('================================')
print('6.1 GridSearchCV results:')
print('6.1.1 Best parameters:')
print(grid.best_params_)
print('6.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('6.2 Test Dataset')
y_pred_rf = grid.predict(X_test)
clfr(y_test, y_pred_rf)
print('================================')
print('6.3 Training Dataset')
y_pred_rf_train = grid.predict(X_train)
clfr(y_train, y_pred_rf_train)

#M7 AdaBoost Classifier
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5), random_state = random_state)
param_grid = {'learning_rate': [0.001, 0.01, 0.1, 1],
              'n_estimators': [10, 100, 200, 500, 1000],
             }
grid = GridSearchCV(adb, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('7 AdaBoost Classifier')
print('================================')
print('7.1 GridSearchCV results:')
print('7.1.1 Best parameters:')
print(grid.best_params_)
print('7.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('7.2 Test Dataset')
y_pred_adb = grid.predict(X_test)
clfr(y_test, y_pred_adb)
print('================================')
print('7.3 Training Dataset')
y_pred_adb_train = grid.predict(X_train)
clfr(y_train, y_pred_adb_train)

#M8 XGBoost Classifier
xgb = XGBClassifier(random_state = random_state)
param_grid = {'max_depth': [3, 4, 5, 6],
              'learning_rate': [0.001, 0.01, 0.1, 1],
              'n_estimators': [10, 100, 200, 500, 1000],
              'subsample': [0.6, 0.7, 0.8, 0.9],
             }
grid = GridSearchCV(xgb, param_grid, cv = cv, scoring = scoring, refit = True)
grid.fit(X_train, y_train)
print('============================================================')
print('8 XGBoost Classifier')
print('================================')
print('8.1 GridSearchCV results:')
print('8.1.1 Best parameters:')
print(grid.best_params_)
print('8.1.2 Best Estimator:')
print(grid.best_estimator_)
print('================================')
print('8.2 Test Dataset')
y_pred_xgb = grid.predict(X_test)
clfr(y_test, y_pred_xgb)
print('================================')
print('8.3 Training Dataset')
y_pred_xgb_train = grid.predict(X_train)
clfr(y_train, y_pred_xgb_train)

#6 Predict validation experiments
y_pred_valexp = grid.predict(X_valexp)
y_pred_pro = grid.predict_proba(X_valexp)
print(y_pred_valexp)
print(y_pred_pro)


























