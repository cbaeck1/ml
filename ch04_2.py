import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격Boston Housing 데이터셋
# 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 1970년대 보스턴 주변의 주택 평균 가격을 예측
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])

# 산점도 : 1개의 특성, 1개의 타겟(숫자)
plt.plot(boston.data[:, 0], boston.target, 'o')
plt.ylim(0, 60)
plt.xlabel("특성1")
plt.ylabel("특성1")
plt.title("boston Scatter Plot")
image.save_fig("boston_Scatter")  
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
# 열의 이름은 boston.feature_names 에 있는 문자열을 사용합니다.
# 사용할 특성의 갯수을 설정
nCase = 4
boston_df = pd.DataFrame(X_train[:,:nCase], columns=range(nCase))
# 데이터프레임을 사용해  특성별 Historgram
boston_df.plot.hist(alpha=0.5)
plt.title("boston Histogram Plot")
image.save_fig("boston_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
if nCase <= 10:
    pd.plotting.scatter_matrix(boston_df, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=2, alpha=.8, cmap=mglearn.cm3)
    plt.title("boston Scatter Plot")
    image.save_fig("boston_Scatter")  
    plt.show()

# 1. k-최근접 이웃 알고리즘 : 분류 
# 2. 선형모델 : 최소제곱
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 훈련 데이터와 테스트 데이터 사이의 이런 성능 차이는 모델이 과대적합되었다는 확실한 신호

# 2. 선형모델 : 릿지
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))

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
mglearn.plots.plot_ridge_n_samples()
plt.title("보스턴 주택가격 데이터셋에 대한 릿지 회귀와 선형 회귀의 학습 곡선")
image.save_fig("boston_learning curve")




# 2. 선형모델 : 라쏘



# 2. 선형모델 : 로지스틱









