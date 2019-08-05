# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_auc_score, roc_curve, precision_recall_curve, f1_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras import utils

#ignore warnings during runing models
import warnings
warnings.filterwarnings('ignore')
random.seed(14)


#2 Load Dataset
#load whole dataset
MONCs = pd.read_excel('MONCs Exp.xlsx').drop(['No'], axis =1)

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
random_state = 14
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify = y, random_state = random_state)
#Normalization 
mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

#load validational data 
MONCs_valexp = pd.read_excel('MONCs ValExp.xlsx').drop(['No'], axis =1)

#define features and target
X_valexp = MONCs_valexp.drop(['P'], axis = 1)

#Normarlization
X_valexp_scaled = mms.transform(X_valexp)

#define classification report
def clfr(y_test, y_pred):
    print('(1) The predictive accuracy is: ', accuracy_score(y_test, y_pred))
    print('(2) Classification report is:\n', classification_report(y_test, y_pred))
    print('(3) Confusion matrix is:\n', confusion_matrix(y_test, y_pred))  

scoring = ['accuracy']

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)


#Creating & compiling a model
model = Sequential()
model.add(Dense(40, input_dim=17, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= scoring)

#Training a model
model.fit(X_train_scaled, y_train, epochs = 300, batch_size = 10)

#Evaluate the model
score = model.evaluate(X_test_scaled, y_test)
print('\nTest loss:', score[0])
print('\nTest accuracy:', score[1] * 100)
print('\n')

#Predict
y_pred_train = model.predict(X_train_scaled)
y_pred_train = np.argmax(y_pred_train, axis = 1)
y_train = np.argmax(y_train, axis =1)
print('ANN-Train')
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

y_pred_test = model.predict(X_test_scaled)
y_pred_test = np.argmax(y_pred_test, axis = 1)
y_test = np.argmax(y_test, axis =1)
print('ANN-Test')
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


#ROC curve of ANN
y_true = y_test
print(y_true)
y_scores = model.predict_proba(X_test_scaled)[:,1]

auc_value = roc_auc_score(y_true, y_scores)
print(auc_value)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of ANN')
plt.legend(loc="lower right")
plt.savefig('ROC curve of ANN')
plt.show()
#PR curve of Logistic
average_precision = average_precision_score(y_true, y_scores)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.figure()
lw = 2
plt.plot(recall, precision, color='darkorange', linewidth=lw, label='PR curve (AP = %0.4f)' % average_precision)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve of ANN')
plt.legend(loc="lower right")
plt.savefig('PR curve of ANN')
plt.show()
np.savetxt("ANN-fpr.txt",fpr)
np.savetxt("ANN-tpr.txt",tpr)
np.savetxt("ANN-precision.txt",precision)
np.savetxt("ANN-recall.txt",recall)













