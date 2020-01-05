# -*- coding: utf-8 -*-
#1 Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve
from sklearn.metrics import f1_score, average_precision_score
from sklearn.utils import shuffle
from sklearn.externals import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras import utils

#ignore warnings during runing models
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
MONCs = pd.read_excel('MONCs Exp.xlsx').drop(['No'], axis =1)

# Split whole dataset into traning and test dataset
X = MONCs.drop(['P'], axis = 1)
y = MONCs['P'].astype('category')

#split dataset into training and test set
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = random_state)
#Normalization 
mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

#define classification report
def clfr(y_test, y_pred):
    print('(1) The predictive accuracy is: ', accuracy_score(y_test, y_pred))
    print('(2) Classification report is:\n', classification_report(y_test, y_pred))
    print('(3) Confusion matrix is:\n', confusion_matrix(y_test, y_pred))  

cv = StratifiedKFold(n_splits = 5, random_state = random_state)
scoring = 'accuracy'

#1 Logistic Regression Classifier
lr = LogisticRegression()
param_grid = {'penalty': ['l2'],
              'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
              'C': [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(lr, param_grid, cv = cv, scoring = scoring, refit = True)
grid_lr.fit(X_train_scaled, y_train)
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
y_pred_lr = grid_lr.predict(X_test_scaled)
clfr(y_test, y_pred_lr)
print('================================')
print('1.3 Training Dataset')
y_pred_lr_train = grid_lr.predict(X_train_scaled)
clfr(y_train, y_pred_lr_train)
joblib.dump(grid_lr, 'grid_lr.pkl')
#1.1 LR ROC curve
y_pred_pro = grid_lr.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns=grid_lr.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#1.2 LR PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#2 Gaussian Naive Bayes Classifier
gnb = GaussianNB()
param_grid = {}
grid_gnb = GridSearchCV(gnb, param_grid, cv = cv, scoring = scoring, refit = True)
grid_gnb.fit(X_train_scaled, y_train)
print('============================================================')
print('2 Gaussian Naive Bayes Classifier')
print('================================')
print('2.1 GridSearchCV results:')
print('2.1.1 Best parameters:')
print(grid_gnb.best_params_)
print('2.1.2 Best Estimator:')
print(grid_gnb.best_estimator_)
print('================================')
print('2.2 Test Dataset')
y_pred_gnb = grid_gnb.predict(X_test_scaled)
clfr(y_test, y_pred_gnb)
print('================================')
print('2.3 Training Dataset')
y_pred_gnb_train = grid_gnb.predict(X_train_scaled)
clfr(y_train, y_pred_gnb_train)
joblib.dump(grid_gnb, 'grid_gnb.pkl')
#2.2 GNB ROC curve
y_pred_pro = grid_gnb.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_gnb.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#2.3 GNB PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#3 k-Nearest Neighbors Classifier
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 11),
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree']}
grid_knn = GridSearchCV(knn, param_grid, cv = cv, scoring = scoring, refit = True)
grid_knn.fit(X_train_scaled, y_train)
print('============================================================')
print('3 k-Nearest Neighbors Classifier')
print('================================')
print('3.1 GridSearchCV results:')
print('3.1.1 Best parameters:')
print(grid_knn.best_params_)
print('3.1.2 Best Estimator:')
print(grid_knn.best_estimator_)
print('================================')
print('3.2 Test Dataset')
y_pred_knn = grid_knn.predict(X_test_scaled)
clfr(y_test, y_pred_knn)
print('================================')
print('3.3 Training Dataset')
y_pred_knn_train = grid_knn.predict(X_train_scaled)
clfr(y_train, y_pred_knn_train)
joblib.dump(grid_knn, 'grid_knn.pkl')
#3.2 KNN ROC curve
y_pred_pro = grid_knn.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_knn.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#3.3 KNN PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#4 Support Vector Machine Classifier
svm = SVC(probability = True)
param_grid = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
              'gamma': ['auto']},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid_svm = GridSearchCV(svm, param_grid, cv = cv, scoring = scoring, refit = True)
grid_svm.fit(X_train_scaled, y_train)
print('============================================================')
print('4 Support Vector Machine Classifier')
print('================================')
print('4.1 GridSearchCV results:')
print('4.1.1 Best parameters:')
print(grid_svm.best_params_)
print('4.1.2 Best Estimator:')
print(grid_svm.best_estimator_)
print('================================')
print('4.2 Test Dataset')
y_pred_svm = grid_svm.predict(X_test_scaled)
clfr(y_test, y_pred_svm)
print('================================')
print('4.3 Training Dataset')
y_pred_svm_train = grid_svm.predict(X_train_scaled)
clfr(y_train, y_pred_svm_train)
joblib.dump(grid_svm, 'grid_svm.pkl')
#4.2 SVM ROC curve
y_pred_pro = grid_svm.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_svm.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#4.3 SVM PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#5 Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=random_state)
param_grid = {'criterion':['gini', 'entropy'], 
              'max_depth': range(3, 11)
             }
grid_dt = GridSearchCV(dt, param_grid, cv = cv, scoring = scoring, refit = True)
grid_dt.fit(X_train_scaled, y_train)
print('============================================================')
print('5 Decision Tree Classifier')
print('================================')
print('5.1 GridSearchCV results:')
print('5.1.1 Best parameters:')
print(grid_dt.best_params_)
print('5.1.2 Best Estimator:')
print(grid_dt.best_estimator_)
y_pred_dt = grid_dt.predict(X_test_scaled)
print('================================')
print('5.2 Test Dataset')
clfr(y_test, y_pred_dt)
y_pred_dt_train = grid_dt.predict(X_train_scaled)
print('================================')
print('5.3 Training Dataset')
clfr(y_train, y_pred_dt_train)
joblib.dump(grid_dt, 'grid_dt.pkl')
#5.2 DT ROC curve
y_pred_pro = grid_dt.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_dt.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#5.3 DT PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#6 Random Forest Classifier
rf = RandomForestClassifier()
param_grid = {'max_depth': range(3, 8),
              'n_estimators': [10, 100, 200, 500, 1000]
             }
grid_rf = GridSearchCV(rf, param_grid, cv = cv, scoring = scoring, refit = True)
grid_rf.fit(X_train_scaled, y_train)
print('============================================================')
print('6 Random Forest Classifier')
print('================================')
print('6.1 GridSearchCV results:')
print('6.1.1 Best parameters:')
print(grid_rf.best_params_)
print('6.1.2 Best Estimator:')
print(grid_rf.best_estimator_)
print('================================')
print('6.2 Test Dataset')
y_pred_rf = grid_rf.predict(X_test_scaled)
clfr(y_test, y_pred_rf)
print('================================')
print('6.3 Training Dataset')
y_pred_rf_train = grid_rf.predict(X_train_scaled)
clfr(y_train, y_pred_rf_train)
joblib.dump(grid_rf, 'grid_rf.pkl')
#6.2 RF ROC curve
y_pred_pro = grid_rf.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_rf.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#6.3 RF PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#7 AdaBoost Classifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
param_grid = {'learning_rate': [0.001, 0.01, 0.1, 1],
              'n_estimators': [10, 100, 200, 500, 1000],
             }
grid_ada = GridSearchCV(ada, param_grid, cv = cv, scoring = scoring, refit = True)
grid_ada.fit(X_train_scaled, y_train)
print('============================================================')
print('7 AdaBoost Classifier')
print('================================')
print('7.1 GridSearchCV results:')
print('7.1.1 Best parameters:')
print(grid_ada.best_params_)
print('7.1.2 Best Estimator:')
print(grid_ada.best_estimator_)
print('================================')
print('7.2 Test Dataset')
y_pred_ada = grid_ada.predict(X_test_scaled)
clfr(y_test, y_pred_ada)
print('================================')
print('7.3 Training Dataset')
y_pred_ada_train = grid_ada.predict(X_train_scaled)
clfr(y_train, y_pred_ada_train)
joblib.dump(grid_ada, 'grid_ada.pkl')
#7.2 ADA ROC curve
y_pred_pro = grid_ada.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_ada.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#7.3 ADA PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#8 XGBoost Classifier
xgb = XGBClassifier(random_state=random_state)
param_grid = {'max_depth': [3, 4, 5, 6],
              'learning_rate': [0.001, 0.01, 0.1, 1],
              'n_estimators': [10, 100, 200, 500, 1000],
              'subsample': [0.6, 0.7, 0.8, 0.9],
             }
grid_xgb = GridSearchCV(xgb, param_grid, cv = cv, scoring = scoring, refit = True)
grid_xgb.fit(X_train_scaled, y_train)
print('============================================================')
print('8 XGBoost Classifier')
print('================================')
print('8.1 GridSearchCV results:')
print('8.1.1 Best parameters:')
print(grid_xgb.best_params_)
print('8.1.2 Best Estimator:')
print(grid_xgb.best_estimator_)
print('================================')
print('8.2 Test Dataset')
y_pred_xgb = grid_xgb.predict(X_test_scaled)
clfr(y_test, y_pred_xgb)
print('================================')
print('8.3 Training Dataset')
y_pred_xgb_train = grid_xgb.predict(X_train_scaled)
clfr(y_train, y_pred_xgb_train)
joblib.dump(grid_xgb, 'grid_xgb.pkl')
#8.2 XGB ROC curve
y_pred_pro = grid_xgb.predict_proba(X_test_scaled)
y_scores = pd.DataFrame(y_pred_pro, columns = grid_xgb.classes_.tolist())[1].values
auc_value = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#8.3 XGB PR curve
average_precision = average_precision_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

#9 Multilayer Perceptron Classifier
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
#Creating & compiling a model
mlp = Sequential()
mlp.add(Dense(40, input_dim=17, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(40, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(2, activation='sigmoid'))
mlp.summary()
mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics= scoring)
mlp.fit(X_train_scaled, y_train, epochs = 300, batch_size = 10)
print('============================================================')
print('9 MLP Classifier')
print('================================')
print('9.1 Test Dataset')
y_pred_mlp = model.predict(X_test_scaled)
y_pred_mlp = np.argmax(y_pred_test, axis = 1)
y_test = np.argmax(y_test, axis =1)
clfr(y_test, y_pred_mlp)
print('================================')
print('9.2 Training Dataset')
y_pred_mlp_train = model.predict(X_train_scaled)
y_pred_mlp_train = np.argmax(y_pred_train, axis = 1)
y_train = np.argmax(y_train, axis =1)
clfr(y_train, y_pred_mlp_train)
#9.2 MLP ROC curve
y_pred_pro = mlp.predict_proba(X_test_scaled)[:,1]
auc_value = roc_auc_score(y_test, y_pred_pro)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro, pos_label=1.0)
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#8.3 MLP PR curve
average_precision = average_precision_score(y_test, y_pred_pro)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_pro)
plt.figure()
plt.plot(recall, precision, color='orange', label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()
