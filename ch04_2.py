import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격 Boston Housing 데이터셋
# 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 1970년대 보스턴 주변의 주택 평균 가격을 예측
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR']+ "\n...")
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])

# 산점도 : 1개의 특성, 1개의 타겟(숫자)
plt.plot(boston.data[:, 0], boston.target, 'o')
plt.ylim(0, 60)
plt.xlabel("특성 CRIM : per capita crime rate by town")
plt.ylabel("Target")
plt.title("boston Scatter Plot")
image.save_fig("boston_Scatter")  
plt.show()

# 히스토그램 : 열의 이름은 boston.feature_names
# 사용할 특성의 갯수을 설정
nCase = 10
boston_df = pd.DataFrame(boston.data[:,:nCase], columns=boston.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
boston_df.plot.hist(alpha=0.5)
plt.title("boston Histogram Plot")
image.save_fig("boston_Histogram")
plt.show() 



# 특성 공학feature engineering : load_extended_boston
# 13개의 원래 특성에 13개에서 2개씩 (중복을 포함해) 짝지은 91개의 특성을 더해 총 104개가 됩니다.
# X, y : numpy ndarray
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
#print("특성 이름:\n{}".format(X.column_names))
print(X, y)

# 훈련 세트, 테스트 세트 random_state=0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 range로 표현
# 사용할 특성의 갯수을 설정
nCase = 4
extended_boston_df = pd.DataFrame(X_train[:,:nCase], columns=range(nCase))
# 데이터프레임을 사용해  특성별 Historgram
extended_boston_df.plot.hist(alpha=0.5)
plt.title("extended_boston_df Histogram Plot")
image.save_fig("extended_boston_df_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
if nCase <= 10:
    pd.plotting.scatter_matrix(extended_boston_df, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=2, alpha=.8, cmap=mglearn.cm3)
    plt.title("extended_boston_df Scatter Plot")
    image.save_fig("extended_boston_df_Scatter")  
    plt.show()

# 1. k-최근접 이웃 알고리즘 : 분류 
# 2. 선형모델 : 최소제곱
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("2. 선형모델 : 최소제곱 훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("2. 선형모델 : 최소제곱 테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 훈련 데이터와 테스트 데이터 사이의 이런 성능 차이는 모델이 과대적합되었다는 확실한 신호

# 2. 선형모델 : 릿지
# 가중치(w) 선택은 훈련 데이터를 잘 예측하기 위해서 뿐만 아니라 추가 제약 조건을 만족시키기 위한 목적도 있습니다.
# 가중치의 절댓값을 가능한 한 작게 만드는 것입니다. 다시 말해서 w의 모든 원소가 0에 가깝게 
# 이를 규제 regularization라고 합니다. 규제란 과대적합이 되지 않도록 모델을 강제로 제한한다는 의미입니다.
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("2. 선형모델 : 릿지 훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("2. 선형모델 : 릿지 테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("2. 선형모델 : 릿지alpha=10 훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("2. 선형모델 : 릿지alpha=10 테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("2. 선형모델 : 릿지alpha=0.1 훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("2. 선형모델 : 릿지alpha=0.1 테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))

# x축의 계수, y 축은 각 계수의 수치
# alpha=10일 때 대부분의 계수는 -3과 3 사이에 위치
# alpha=1일 때 Ridge 모델의 계수는 좀 더 커지고
# alpha=0.1일 때 계수는 더 커지며 
# 아무런 규제가 없는(alpha=0) 선형 회귀의 계수는 값이 더 커져 그림 밖으로 넘어갑니다.

plt.figure(figsize=(14, 8))
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.title("선형 회귀와 몇 가지 alpha 값을 가진 릿지 회귀의 계수 크기 비교")
image.save_fig("boston_Ridge_coef")
plt.show() 

# 규제의 효과를 이해하는 또 다른 방법은 alpha 값을 고정하고 훈련 데이터의 크기를 변화시켜 보는 것입니다. 
# 보스턴 주택가격 데이터셋에서 여러 가지 크기로 샘플링하여 LinearRegression과 Ridge(alpha=1)을 적용
# 데이터셋의 크기에 따른 모델의 성능 변화를 나타낸 그래프를 학습 곡선learning curve이라고 합니다
# 릿지에는 규제가 적용되므로 릿지의 훈련 데이터 점수가 전체적으로 선형 회귀의 훈련 데이터 점수보다 낮습니다.
# 그러나 테스트 데이터에서는 릿지의 점수가 더 높으며 특별히 작은 데이터셋에서는 더 그렇습니다.
# 데이터셋 크기가 400 미만에서는 선형 회귀는 어떤 것도 학습하지 못하고 있습니다.
plt.figure(figsize=(14, 8))
mglearn.plots.plot_ridge_n_samples()
plt.title("보스턴 주택가격 데이터셋에 대한 릿지 회귀와 선형 회귀의 학습 곡선")
image.save_fig("boston_learning curve")
plt.show() 

# 2. 선형모델 : 라쏘
# 계수를 0에 가깝게, L1 규제
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
# 과소적합이며 105개의 특성 중 4개만 사용
print("2. 선형모델 : 라쏘 훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("2. 선형모델 : 라쏘 테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("2. 선형모델 : 라쏘 사용한 특성의 수: {}".format(np.sum(lasso.coef_ != 0)))

# "max_iter" 기본값을 증가시키지 않으면 max_iter 값을 늘리라는 경고가 발생합니다.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso001.coef_ != 0)))

# alpha 값을 너무 낮추면 규제의 효과가 없어져 과대적합이 되므로 LinearRegression의 결과와 비슷
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso00001.coef_ != 0)))

# alpha=1      계수 대부분이 0일 뿐만 아니라 나머지 계수들도 크기가 작다는 것을 알 수 있습니다.
# alpha=0.01   대부분의 특성이 0이 되는 (정삼각형 모양으로 나타낸) 분포를 얻게 됩니다. 
# alpha=0.0001 계수 대부분이 0이 아니고 값도 커져 꽤 규제받지 않은 모델을 얻게 됩니다. 
# alpha=0.1인 Ridge 모델은 alpha=0.01인 라쏘 모델과 성능이 비슷하지만 Ridge를 사용하면 어떤 계수도 0이 되지 않습니다.

plt.figure(figsize=(14, 8))
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.legend()
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.title("릿지 회귀와 alpha 값이 다른 라쏘 회귀의 계수 크기 비교")
image.save_fig("boston_lasso_coef")
plt.show() 


# 2. 선형모델 : 로지스틱 
# 예측한 값을 임계치 0과 비교 0보다 작으면 클래스를 -1이라고 예측하고 0보다 크면 +1이라고 예측
# 분류용 선형 모델에서는 결정 경계가 입력의 선형 함수

# 2. 선형모델 : 서포트 벡터 머신 




