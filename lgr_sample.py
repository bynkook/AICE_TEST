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
print(y_pred[0:5])  # [1 2 2 0 0]

print('confusion matrix = \n', confusion_matrix(y_true=y_test, y_pred=y_pred))
print('accuracy = ', accuracy_score(y_true=y_test, y_pred=y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred))    # 멀티클래스 분류문제도 잘 계산한다

# %%

# roc auc score 계산
print(lr.predict_proba(X_test)[0:5]) # [[3.88894905e-02 8.00353277e-01 1.60757233e-01], ... ]

y_prob_score1 = lr.predict_proba(X_test)[:,0]   # y=0 일 확률
y_prob_score2 = lr.predict_proba(X_test)[:,1]   # y=1 일 확률
y_prob_score3 = lr.predict_proba(X_test)[:,2]   # y=2 일 확률

# ROC_AUC_SCORE: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters).
print('roc_auc_score for label 1 = ', roc_auc_score(y_true = (y_test == 0).astype(int), y_score=y_prob_score1))    # y_true <= y==0 일때만 1
print('roc_auc_score for label 2 = ', roc_auc_score(y_true = (y_test == 1).astype(int), y_score=y_prob_score2))    # y_true <= y==1 일때만 1
print('roc_auc_score for label 3 = ', roc_auc_score(y_true = (y_test == 2).astype(int), y_score=y_prob_score3))    # y_true <= y==1 일때만 1

# 멀티클래스 문제도 한번에 계산되는지 이것저것 테스트
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average=None, multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='micro', multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='macro', multi_class='ovr'))
print('roc_auc_score = ', roc_auc_score(y_test, lr.predict_proba(X_test), average='macro', multi_class='ovo'))

# %%

'''
#### Target y 값이 (0,1) 같이 2개 레이블(False/True)일 경우의 일반적인 절차 ####

# 데이터 세트 분리 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 확률 계산
y_prob_score = model.predict_proba(X_test)[:, 1]

# ROC 곡선 계산
fpr, tpr, _ = roc_curve(y_test, y_prob_score)

# AUC 계산
roc_auc = auc(fpr, tpr)

# ROC 곡선 그리기
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
'''