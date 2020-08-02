import pandas as pd
import numpy as np
import mglearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 데이터셋을 만듭니다. n_samples = 400
X, y = mglearn.datasets.make_wave(n_samples=400)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 산점도 : 2개의 특성으로
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel("타깃")
plt.title("Make Wave Scatter Plot")
image.save_fig("Make_Wave_Scatter")  
plt.show()

# ŷ = w[0] × x[0] + b
# 1차원 wave 데이터셋으로 파라미터 w[0]와 b를 직선이 되도록 학습

mglearn.plots.plot_linear_regression_wave()
image.save_fig("Make_Wave_regression")  
plt.show()

# 선형 회귀(최소제곱법)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 선형 회귀는 예측과 훈련 세트에 있는 타깃 y 사이의 평균제곱오차(mean squared error)를 최소화하는 파라미터 w와 b를 찾습니다. 
# 평균제곱오차는 예측값과 타깃값의 차이를 제곱하여 더한 후에 샘플의 개수로 나눈 것입니다. 
# 선형 회귀는 매개변수가 없는 것이 장점이지만, 그래서 모델의 복잡도를 제어할 방법도 없습니다.
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
# 기울기 파라미터(w)는 가중치weight 또는 계수coefficient라고 하며 lr 객체의 coef_ 속성
# 편향offset 또는 절편intercept 파라미터(b)는 intercept_ 속성이다
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
















