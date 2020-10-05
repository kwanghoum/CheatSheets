### 피처 엔지니어링

train, test 묶어서 한번에

```python
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
display(train,test)
all_data = pd.concat([train, test])

# train셋의 y값과 필요없는칼럼(문자열?) 지우기
all_data2 = all_data.drop(["bidder_id","outcome"],1)

# 후에 모델 학습전에 분리
train2 = all_data[:len(train)]
test2 = all_data[len(train):]
```

결측값

```python
all_data = all_data.fillna(-1)
```

문자열칼럼이 값들이 겹치는 범주형칼럼인지 확인

```python
all_data["payment_account"].nunique()
```

datetime

```python
all_data["datetime"] = all_data["datetime"].astype("datetime64")
all_data["hour"] = all_data["datetime"].dt.hour
all_data["day"] = all_data["datetime"].dt.day
all_data["weekday"] = all_data["datetime"].dt.weekday
all_data["month"] = all_data["datetime"].dt.month
all_data["year"] = all_data["datetime"].dt.year
```

그래프그려서 써도 괜찮을 칼럼인지

```python
import matplotlib.pyplot as plt #밑그림을 그릴때(몇개를 어디에 )
import seaborn as sns           #어떤 그래프를 그릴지 결정

plt.figure(figsize = (20,12)) #figsize 가로세로 길이
sns.boxplot(all_data["Type"],all_data["Weekly_Sales"],showfliers=False) #showfliers:결측값
# 새로운 파일을 all_data에 merge한 후 새로 추가된 'Type'칼럼이 쓸만한지 확인하는 과정
```

라벨인코딩

```python
# 칼럼 하나 직접 라벨인코딩
all_data = all_data.replace({"A": 1, "B": 2, "C": 3}) #'type'칼럼의 항목 A,B,C를 직접 숫자로 바꿔줌

# 칼럼 두개만 간단하게 라벨인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_data2["address"] = le.fit_transform(all_data2["address"])
all_data2["payment_account"] = le.fit_transform(all_data2["payment_account"])

# 여러 범주형칼럼들(문자형)을 한번에 라벨인코딩
all_data["Make"].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_data["Make"] = le.fit_transform(all_data["Make"]) #fit함수(숫자로 등록)와 transform함수(등록된걸 바탕으로 변환)를 동시에
le.classes_  #이거 앞에서 부터 0번
cate_cols = all_data.columns[all_data.dtypes == 'object']
for i in cate_cols:
    all_data[i] = le.fit_transform(list(all_data[i]))  #결측치가 형식이 float이어서 오류날 수 있음 -> list로 묶어버림
```

원핫인코딩

```python
# 선형모델 쓸 때 레이블인코딩 쓰면 안됨.(모델이 범주적인 숫자를 그 크기에 의미를 부여해버림)
# 이때(거리기반의 학습 쓸 때)는 원핫인코딩 써야함
all_data3 = pd.get_dummies(all_data2) #데이터타입 object인 칼럼에만 접근해서 원핫인코딩 해줌.
# 원핫인코딩 쓰면 칼럼갯수 많아지긴 하는데 한 백몇개칼럼까지는 커버치긴함
# (이건 경험적으로 내가 많이 해보고 알아야함)
# 데이터의 갯수에 비해 칼럼갯수
```

칼럼많아서 안보이는거 있을때 다 보기(row도)

```python
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 99
```

regularization(scaling)

```python
#regularization     https://dailyheumsi.tistory.com/57
#과적합될것같으면 weight를 0가까이 해버리는것
#선형모델에는 규제라는 하이퍼파라메터가 있음

#scaling   : 거리기반으로 학습하는 모델에서는 스케일링 꼭 해줘야함

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
all_data4 = ss.fit_transform(all_data3)  #데이터프레임을 array로 바꿔버림 이따 다시 바꿔주자
all_data4 = pd.DataFrame(all_data4, columns=all_data3.columns) #array를 다시 데이터프레임으로
```

### 모델링

랜덤포레스트

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=4)

# %%time
rf.fit(train2,np.log(train["count"]))
result = rf.predict(test2) # 분류문제에선 rf.predict_proba()
```

xgboost

```python
from xgboost import XGBRegressor
xgb = XGBRegressor(learning_rate=0.1)

xgb.fit(train2, train["SalePrice"])
result2 = xgb.predict(test2)
```

catboost

lgbm

```python
from lightgbm import LGBMRegressor
lgb = LGBMRegressor(n_estimators) #num_leaves

lgb.fit(train2,train['Weekly_Sales'])
result = lgb.predict(test2)
```

ridge

```python
from sklearn.linear_model import Ridge #L2규제 활용한 모델
rg = Ridge(alpha=700)
rg.fit(train2, train["SalePrice"])
result = rg.predict(test2)
```

앙상블

```python
sub = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")
sub["SalePrice"] = np.exp(result)*0.5 + result2*0.5
sub.to_csv("sub0918.csv",index=0)
```

### 기타

로그함수(y값이 편향돼있을때)

```python
# y값이 편향되있는지 ㄱㅊ한지 확인
import matplotlib.pyplot as plt #밑그림을 그릴때(몇개를 어디에 )
import seaborn as sns
plt.figure(figsize = (20,12))
sns.distplot(all_data["count"]) #y축은 상대적인 빈도수
#보니까 특징 : 데이터가 한쪽에 쏠려있음, 오른쪽에 극단값(outlier)가 있음(꼬리가 김 1000...)

# 모델학습할 때 로그씌워주고,
rf.fit(train2,np.log(train["count"]))
# 후에 result값 담을 때, 다시 빼줌
sub["count"] = np.exp(result)
```

코릴레이션 (어떤 모델 쓸지 결정하는 과정에서 함찍어보기)

```python
train.corr()["SalePrice"].sort_values(ascending=False) #값이 0.3보다만 높아도 엄청 높은 거임
# 보통 이렇게 코릴레이션 높은거 많이 없음
# 이렇게 코릴레이션이 높은게 많으면 선형모델 써보는게 좋음
```

교차검증

```python
from sklearn.model_selection import cross_val_score
np.sqrt(-cross_val_score(rg, train2, train["SalePrice"], n_jobs=4, cv=10, #cv기본값은 5(train셋을 5등분)
													scoring='neg_mean_squared_error', ).mean())
for i in [1, 10, 20, 50, 200, 300, 500, 700, 1000]:
    rg = Ridge(alpha=i)
    print(np.sqrt(-cross_val_score(rg, train2, train["SalePrice"], n_jobs=4, cv=10,
																		scoring='neg_mean_squared_error', ).mean()))
# 어떤 alpha값이 젤 좋을지 결정했고 그 후
rg = Ridge(alpha=700)
rg.fit(train2, np.log(train["SalePrice"]))
result = rg.predict(test2)
```

부가적인 데이터셋 활용법(merge)

```python
store = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
all_data = pd.merge(all_data, store, on = "Store", how = "left") #merge는 딱 한번만 실행해야함. 또하면 또만들어짐
```

데이터 후처리

```python
#미국 현충일
sub.iloc[1258:1269, 1]= sub.iloc[1258:1269, 1]*0.5
sub.iloc[4492:4515, 1]= sub.iloc[4492:4515, 1]*0.5
#크리스마스 이브
sub.iloc[6308:6330, 1]= sub.iloc[6308:6330, 1]*0.5
sub.iloc[3041:3063, 1]= sub.iloc[3041:3063, 1]*0.5
#크리스마스
sub.iloc[6332:6354, 1]= sub.iloc[6332:6354, 1]*0.5
sub.iloc[3065:3087, 1]= sub.iloc[3065:3087, 1]*0.5
#추수감사절
sub.iloc[5992:6015, 1]= sub.iloc[5992:6015, 1]*0.5
sub.iloc[2771:2794, 1]= sub.iloc[2771:2794, 1]*0.5
```

submission파일 만들기

```python
sub = pd.read_csv("/kaggle/input/DontGetKicked/example_entry.csv")
sub["IsBadBuy"] = result[:,1]
sub.to_csv("sub0913.csv", index=0)
```

모델 학습결과 어떤 칼럼을 중요하게 여겼나 보는법

```python
pd.options.display.max_rows = 99
pd.Series(xgb.feature_importances_, index=train2.columns).sort_values(ascending=False)
```

groupby()함수

```python
# 예시1
bids.groupby("bidder_id")["merchandise"].nunique()
# 예시2
nunique = bids.groupby("bidder_id")["auction", "device", "country"].nunique()
all_data = pd.merge(all_data, nunique, on = "bidder_id", how = "left")
#예시3
datetime_mean = bids.groupby("bidder_id")["year","month","day"].mean()
all_data = pd.merge(all_data, datetime_mean, on = "bidder_id", how = "left")
```

apply()함수

```python
# 예시1
bids["datetime"] = bids["time"].apply(lambda x : datetime.fromtimestamp(x / 7000000))
# 예시2
all_data["address"] = all_data["address"].apply(lambda x : x[:5])
```

결과값이 어떻게 분포돼있나 쳌

```python
train["outcome"].value_counts()
```

랜덤시드고정

```python
rf = RandomForestClassifier(n_jobs=4, random_state=1234) #random_state 랜덤시드 고정
```

베이스라인

```python
# 전처리
train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")
all_data = pd.concat([train, test])

all_data2 = all_data.drop(["Date","Weekly_Sales"],1)
train2 = all_data2[:len(train)]
test2 = all_data2[len(train):]
# train2 = train.drop(['Date','Weekly_Sales'],1)
# test2 = test.drop('Date',1)

# 모델링
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=4)
# %%time
rf.fit(train2,train['Weekly_Sales'])
result = rf.predict(test2)

# 결과 정리&제출
sub = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")
sub["Weekly_Sales"] = result
sub.to_csv("sub_0824.csv",index = 0)
```
