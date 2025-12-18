# %%
import pandas as pd

# names = 로 컬럼명을 순서대로 지정할 수 있다.
df = pd.read_csv('grouplens/users.dat', sep='::', names=['id','gender','age','occupation','region'])

# join 기본값은 outer, ignore_index=False 이므로 둘중 하나만 데이터가 있어도 기본적으로 결합됨(NaN 발생)
# ignore_index=True 는 인덱스 재생성(0부터)
# concat 은 인덱스 사용해서 합치는 함수
pd.concat([A, B], ignore_index=True, axis=1) # 수평 union
pd.concat([A, B], ignore_index=True, axis=0) # 수직 union
pd.concat([A, B], ignore_index=True, axis=1, join='inner')

# how='left', 'right', 'outer' 가능하나 결측치가 발생하므로 실제 시험에는 출제 안함.
pd.merge(A, B, how='inner', on='item_id')	# 교집합 없으면 Null(empty)
df1.merge(df2, how='inner', on=['col1','col2'])	# 다중 컬럼으로 merge
df1.merge(df2, how='inner', left_on='이름', right_on='성명') # 좌 우 컬럼명이 다를때

# 실습
df.loc[ [1,3,5], ['col1','col3'] ] 1~5 행과 col1, col3 컬럼에 해당하는 table 을 가져옴
df.loc[ 0:20, 'col1':'col3' ] 0~20번 인덱스의 행(총 21개), col1~col3 까지 컬럼만 출력
df2 = df.copy() df2를 수정해도 df는 영향 없음(deep copy)

# annual_income 최저값, 최고값 탐색 (boolean indexing)
print(df[df.annual_income == df.annual_income.min()])
idx_min = df[df.annual_income == df.annual_income.min()].index[0]
또는
idx_min = df.annual_income.idxmin()
# 최소값 인덱스 확인해서 최소값 있는 행 삭제
# idxmin : Return index of first occurrence of minimum over requested axis.  NA/null values are excluded.
df = df.drop(idx_min, axis=0)

# 결측치 처리
df.info() 를 보고 결측치 갯수를 파악가능
df.isnull().head() True, False 로 출력(처음 5개만)
df.isnull().sum() 컬럼별 True 갯수 출력

df.fillna(0) 결측치 0으로 채움
df.interpolate() 앞뒤 선형보간값 채움

# 결측치 처리 - listwise, pairwise 처리 방법 구분
df.dropna() 행 삭제(전체 컬럼 탐색)
df.dropna(how='all') 행 전체 값이 null 인 경우 그 행을 삭제
df.dropna(subset='col3') col3 만 탐색해서 null 인 행 삭제
df.dropna(subset=['col1', 'col2']) col1, col2 만 탐색해서 null 인 행 삭제
df.isna().sum() 결측치 처리한 결과 확인

# 결측치 대체
df.score.fillna(0)
df['Age'].fillna(df['Age'].mean())
df['Age'].fillna(df['Age'].median())
mode_val = df['Age'].mode()[0]  # [0] 첫번째 값을 가져와야 한다! 주의!
df['Age'].fillna(mode_val)

df.drop('col1', axis=1, inplace=True) 컬럼 삭제
mean = df['col3'].mean() 평균값 계산(null 값 제외한 산술평균)
df['col3'] = df['col3'].fillna(mean)

df.reset_index(drop=True, inplace=True) 이빨빠진 기존 인덱스를 버리고 새로 생성, df 업데이트
df.to_csv('output.csv', index=False) 인덱스를 제외한 모든 컬럼을 .csv 파일에 저장

# 범주형 최빈값 찾아서 채우기
df['col4'].value_counts() 로 보고 범주형 데이터 갯수 파악
df['col4'].fillna('val') val = 최빈값 텍스트형

# 이상치 처리하기 실습
output = df[ df['class'] != 'H' ]	# boolean indexing 으로 class = H 를 제거
df4['col1'] = df4['col1'].replace('cc','DD')	# replace로 col1 의 값 교체

# 이상치 확인
q1 = df['col'].quantile(0.25) # .quantile()은 Series 객체에서, .quantiles()는 DataFrame 객체에서 사분위수 계산
q3 = df['col'].quantile(0.75)
upper_bound = q3 + 1.5 * IQR
lower_bound = q1 - 1.5 * IQR
IQR = q3 - q1
# 'col' 컬럼에서 1.5 * IQR 보다 작거나 큰 값은 이상치로 간주
outliers = df[ (df['col'] < lower_bound) | (df['col'] > upper_bound) ]
# 이상치 제거 (이상치 있는 행 전체 삭제)
df = df[ (df['col'] >= lower_bound) & (df['col'] <= upper_bound) ]
# 이상치 대치 (컬럼에 이상치 값이 있으면 중앙값으로 대치) - 이것은 시험에 꼭 나온다! 
# .loc[row_index, column_index] 사용법 : 레이블로 인덱스를 지정한다!
df.loc[ (df.col < lower_bound) | (df.col > upper_bound), 'col'] = df.annual_income.median()	# 이상치 대치

# 구간지정해서 범주화 (Binning)
df['level'] = pd.cut( df['col'], bins=[0, q1, q3, df['col'].max()], labels=['low', 'mid', 'high'] )		# 분할 구간 지정
df['level'] = pd.qcut( df['col'], 3, labels=['low', 'mid', 'high'] )		# 등순위 분할(갯수 동일)
# bins = [0,10,20,30,40,50] 이런식으로 설정할 수 있다.
##### right : Indicates whether bins includes the rightmost edge or not.
# If right == True (the default), then the bins [1, 2, 3, 4] indicate (1,2], (2,3], (3,4].
# This argument is ignored when bins is an IntervalIndex.


# 값 보기
df.head()
df['col'].value_counts()
df.isna().sum()

# Scaling : standardization (정규화)
# Z-score 정규화
num_df = df.select_dtypes(include='number')
print((num_df - num_df.mean())/num_df.std())
# sklearn
scaler = StandardScaler()
df3['StdScaled'] = scaler.fit_transform(df3[['score']]) # [['...']] 는 Series 가 들어가면 error이기 때문에 DataFrame 을 넘겨야 한다.


# 숫자 컬럼명, 문자열 컬럼명 *리스트*로 저장
# 'number' 'bool', 'boolean' : numpy 타입
# object, bool : generic 타입
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include=object).columns.tolist()
bool_cols = df.select_dtypes(include=bool).columns.tolist()
num_cols.remove('y_target')  # target 컬럼명 삭제

# GridSearchCV
하이퍼 파라메터를 순차적으로 입력해 학습을 하고 측정을 하면서
가장 좋은 최적의 파라미터를 알려준다.
from sklearn.model_selection import GridSearchCV
rfc = RandonForestClassifier()
params = {'n_estimators':[100, 150], 'max_depth':[2, 5]}
grid_rfc = GridSearchCV(rfc,  param_grid = params)
grid_rfc.fit(X_train, y_train)