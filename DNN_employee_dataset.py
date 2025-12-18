# %%
import pandas as pd
import re

# CSV 파일 읽기
df = pd.read_csv('data/Employee_Complete_Dataset.csv')

# True/False, Yes/No 컬럼을 판별하여 Boolean 으로 치환
def replace_boolean(df):
    for col in df.columns:
        # 컬럼이 object 타입
        if df[col].dtype == 'object':
            # 대소문자 구분 없이 True/False, Yes/No 문자열을 True/False Boolean 으로 변경
            lower_col = df[col].astype(str).str.lower()
            unique_vals = lower_col.unique()
            if set(unique_vals) & set(['true', 'false', 'yes', 'no']):                
                df[col] = lower_col.map({'true': True, 'false': False, 'yes': True, 'no': False})
    return df

# True/False, Yes/No 치환
replace_boolean(df)
print(df.info())

# %%

# Target 컬럼 생성
df['Job_Satisfied'] = df['Job_Satisfaction'] > 4.5

# 불필요한 컬럼 제거
df = df.drop(['Employee_number','Employee_name','Education_level','Department','Role','is_outlier','Job_Satisfaction'], axis=1)

# 결측치 제거(map 함수로 변경되지 않은 None 값 있는 행도 삭제된다)
df = df.dropna(axis=0, how='any')

# object 컬럼
object_cols = [col for col in df.columns if df[col].dtype == 'object']

# True/False 컬럼을 제외
categorical_cols = []
for col in object_cols:
    if not df[col].astype(str).str.lower().isin(['true', 'false', 'yes', 'no', '1', '0']).all():
        print(f'Categorical column detected: {col}')
        categorical_cols.append(col)

# 컬럼 값 소문자로 변환 후 get_dummies()
for col in categorical_cols:
    df[col] = df[col].astype(str).str.lower()
df = pd.get_dummies(df, columns=categorical_cols)

# 입력데이터 확인
print(df.info())
print(df.head())
df.to_csv('Employee_processed.csv', encoding='utf-8-sig', index=False)

#%% 모델링

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 2. 타겟과 피처 분리
X = df.drop('Job_Satisfied', axis=1)
y = df['Job_Satisfied']

# 3. 타겟 원핫 인코딩 (softmax 2개 출력용)
y = to_categorical(y, num_classes=2)

# 4. train/test 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(y_train.shape)
print(y_val.shape)
print(y_train[0:5])

# %%

# 5. 모델 생성
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 콜백 정의
estop = EarlyStopping(monitor='val_loss', patience=10)
mcheck = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# 7. 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=16,
          validation_data=(X_val, y_val),
          callbacks=[estop, mcheck])

print(model.summary())

# %% 새로운 데이터 예측

# 새로운 데이터 읽기 및 동일 전처리
new_df = pd.read_csv('data/Employee_new.csv')

# True/False, Yes/No 치환
new_df = replace_boolean(new_df)

# 불필요한 컬럼 제거
new_df = new_df.drop(['Employee_number','Employee_name','Education_level','Department','Role','is_outlier'], axis=1)

# 명목형 컬럼 처리(소문자 변환 및 get_dummies)
for col in categorical_cols:
    if col in new_df.columns:
        new_df[col] = new_df[col].astype(str).str.lower()

'''
머신러닝 모델에 안정적인 입력 제공을 위해 꼭 필요한 전처리 단계로,
만약 향후 여러 데이터셋 컬럼 차이가 자주 발생한다면 pandas의 align() 함수도 고려할 수 있지만,
현재 방식처럼 컬럼 직접 검사와 0 채우기가 직관적이고 함께 사용하기 쉽다.
'''

# 원핫 인코딩 적용 (train 데이터와 컬럼이 일치하도록 리인덱싱)
new_df = pd.get_dummies(new_df, columns=[col for col in categorical_cols if col in new_df.columns])

# train 데이터 컬럼과 맞추기 위해 없는 컬럼은 0으로 채움
for col in X.columns:
    if col not in new_df.columns:
        new_df[col] = False

# 순서 맞추기
new_df = new_df[X.columns]  # 여기서 y 컬럼은 제외된다

# 입력데이터 확인
print(new_df.info())
print(new_df.head())

# 예측
pred_probs = model.predict(new_df)
print(pred_probs)

pred_labels = np.argmax(pred_probs, axis=1)
print(pred_labels)

pred_divorced_earlier = ['Yes' if x == 1 else 'No' for x in pred_labels]
print(pred_divorced_earlier)