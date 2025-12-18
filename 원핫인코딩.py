# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
df_load = pd.read_csv('9_loan_approval.csv')

# %%
df = df_load.loc[0:30, ['employment_type', 'annual_income', 'loan_purpose', 'home_ownership']].copy()
df = df.dropna()
# 주의 : .loc로 row 인덱스 슬라이싱할때 0:20 은 0~20번 인덱스까지 포함(총 21개 행)
# 주의 : [0:20] 는 문법이 틀리므로 오류임. 리스트는 [0,1,2,3,4,...20] 이렇게 써야함
#        즉, df_load.loc[ [0:20], ['emp...]] 는 오류임.

# 범주형 변수 값 종류 확인
print(df.employment_type.unique())
print(df.loan_purpose.unique())
print(df.home_ownership.unique())
print(df.head())
print(df.tail())

# %%
df_new = df_load.loc[40:45, ['employment_type', 'annual_income', 'loan_purpose', 'home_ownership']].copy()
df_new = df_new.dropna()

# 범주형 변수 값 종류 확인
print(df_new.employment_type.unique())
print(df_new.loan_purpose.unique())
print(df_new.home_ownership.unique())
print(df_new)

# %%
# 원핫인코딩
df_enc1 = pd.get_dummies(df)
df_enc2 = pd.get_dummies(df_new)

# 컬럼명 비교
print(df_enc1.columns)
print(df_enc2.columns)
print(set(df_enc1.columns) - set(df_enc2.columns))

# 자영업, 무직
# 주택, 생활비
# 값이 df_new 에 없기 때문에 원핫 인코딩 후 컬럼 불일치 발생
# 이렇게 되면 훈련된 DNN 모델로 새로운 데이터로 predict 할때 오류가 발생하므로
# 새로운 데이터를 가져와서 원핫인코딩 할때는
# 없는 컬럼명은 새로 생성하고 False 로 채워줘야 한다.
# 또한, 컬럼 순서도 훈련데이터셋과 동일하게 맞춰줘야 한다.
# 시험이 쉬우면, new_data 를 컬럼 맞춰서 인코딩된걸 수험자가 로드만 하게 제공하고
# 시험이 어려우면, 수험자가 new_data 로드 하고 인코딩하고 컬럼을 맞춰야 한다.

# %% 없는 컬럼명 생성하여 False 로 채우고 컬럼 순서 맞춤을 한줄로 끝낸다
df_enc2_fixed = df_enc2.reindex(columns=df_enc1.columns, fill_value=False)

# 컬럼명 비교
print(df_enc1.columns)
print(df_enc2_fixed.columns)
col_name_set = set(df_enc1.columns) - set(df_enc2_fixed.columns)
print(col_name_set)
# %%
