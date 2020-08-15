import pandas as pd
import numpy as np
import mglearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 5. 선형 회귀(최소제곱법)을 위한 wave 데이터셋. n_samples = 40
X, y = mglearn.datasets.make_wave(n_samples=40)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

# ŷ = w[0] × x[0] + b
# 1차원 wave 데이터셋으로 파라미터 w[0]와 b를 직선이 되도록 학습
mglearn.plots.plot_linear_regression_wave()
plt.title('wave 데이터셋에 대한 선형 모델의 예측')
image.save_fig("Make_Wave_regression")  
plt.show()

# 특성이 많은 데이터셋이라면 선형 모델은 매우 훌륭한 성능을 낼 수 있습니다. 
# 특히 훈련 데이터보다 특성이 더 많은 경우엔 어떤 타깃 y도 완벽하게 (훈련 세트에 대해서) 선형 함수로 모델링할 수 있습니다
# 선형모델 : 최소제곱, 릿지, 라쏘, 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 2. 선형모델 : 최소제곱
# 선형 회귀는 예측과 훈련 세트에 있는 타깃 y 사이의 평균제곱오차(mean squared error)를 최소화하는 파라미터 w와 b를 찾습니다. 
# 평균제곱오차는 예측값과 타깃값의 차이를 제곱하여 더한 후에 샘플의 개수로 나눈 것입니다. 
# 선형 회귀는 매개변수가 없는 것이 장점이지만, 그래서 모델의 복잡도를 제어할 방법도 없습니다.
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
# 기울기 파라미터(w)는 가중치weight 또는 계수coefficient라고 하며 lr 객체의 coef_ 속성
# 편향offset 또는 절편intercept 파라미터(b)는 intercept_ 속성이다
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("2. 선형모델 : 최소제곱 훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("2. 선형모델 : 최소제곱 테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))

########################################################################
# 2. 선형모델 : 릿지
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

# alpha 값과 모델 복잡도의 관계
# 좋은 매개변수를 선택하는 방법은 5장에서 
# alpha 값과 coef_ 속성과의 관계
# 높은 alpha 값은 제약이 더 많은 모델이므로 작은 alpha 값일 때보다 coef_의 절댓값 크기가 작을 것이라고 예상할 수 있습니다
# x 축은 coef_의 원소를 위치대로 나열한 것입니다. 
# 즉 x=0은 첫 번째 특성에 연관된 계수이고 x=1은 두 번째 특성에 연관된 계수입니다. 이런 식으로 x=100까지 계속됩니다. 
# y 축은 각 계수의 수치를 나타냅니다. alpha=10일 때 대부분의 계수는 -3과 3 사이에 위치
# alpha=1일 때 Ridge 모델의 계수는 좀 더 커졌습니다. 
# alpha=0.1일 때 계수는 더 커지며 아무런 규제가 없는(alpha=0) 선형 회귀의 계수는 값이 더 커져 그림 밖으로 넘어갑니다

plt.figure(figsize=(14, 8))
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-3, 3)
plt.legend()
plt.title('선형 회귀와 몇 가지 alpha 값을 가진 릿지 회귀의 계수 크기 비교')
image.save_fig("Make_Wave_regression_ridge")  
plt.show()

# 학습곡선















