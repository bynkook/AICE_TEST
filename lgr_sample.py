# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score

# %%

iris_df = pd.read_csv('data/iris.csv')
print(iris_df.head())
print(iris_df.info())

# %%

# sns.pairplot(iris_df, hue='species')
# plt.show()

# %% Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])

print(iris_df.head())

# %%

X = iris_df.drop(columns='species')
y = iris_df['species']

print(y.value_counts())    # y값은 '0, 1, 2' 3종류임 (멀티클래스 분류)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026)

rs = RobustScaler()
X_train = rs.fit_transform(X_train)
X_test = rs.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)

# %%

print("coefficient = ", lr.coef_)
print("intercept = ", lr.intercept_)

y_pred = lr.predict(X_test)
print(y_pred[0:5])

print('confusion matrix = \n', confusion_matrix(y_true=y_test, y_pred=y_pred))
print('accuracy = ', accuracy_score(y_true=y_test, y_pred=y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred))  # 멀티클래스 분류문제도 잘 계산한다

# %%

# roc auc score 계산
y_prob_score1 = lr.predict_proba(X_test)[:,0]
y_prob_score2 = lr.predict_proba(X_test)[:,1]
y_prob_score3 = lr.predict_proba(X_test)[:,2]

print('roc_auc_score for label 1 = ', roc_auc_score(y_true = (y_test == 0).astype(int), y_score=y_prob_score1))
print('roc_auc_score for label 2 = ', roc_auc_score(y_true = (y_test == 1).astype(int), y_score=y_prob_score2))
print('roc_auc_score for label 3 = ', roc_auc_score(y_true = (y_test == 2).astype(int), y_score=y_prob_score3))

print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average=None, multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='micro', multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='macro', multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='macro', multi_class='ovo'))