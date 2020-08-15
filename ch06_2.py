import pandas as pd
import numpy as np
import mglearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 6. 세 개의 클래스를 가진 간단한 blobs 데이터셋
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 산점도
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1", "클래스 2"])
plt.title("세 개의 클래스를 가진 2차원 데이터셋")
image.save_fig("blobs_Scatter")  
plt.show()

########################################################################
# 2. 선형모델 : 서포트 벡터 머신 
# 이진 분류 알고리즘을 다중 클래스 분류 알고리즘으로 확장하는 보편적인 기법은 일대다방법
# 일대다 방식은 각 클래스를 다른 모든 클래스와 구분하도록 이진 분류 모델을 학습
# 결국 클래스의 수만큼 이진 분류 모델이 만들어집니다. 
# 예측을 할 때 이렇게 만들어진 모든 이진 분류기가 작동하여 가장 높은 점수를 내는 분류기의 클래스를 예측값으로 선택
# 세 개의 클래스를 가진 간단한 데이터셋에 일대다 방식을 적용
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
print("계수 배열의 크기: ", linear_svm.coef_.shape)
print("절편 배열의 크기: ", linear_svm.intercept_.shape)

# 결정경계 : 세 개의 이진 분류기가 만드는 경계
plt.figure(figsize=(14, 8))
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(['클래스 0', '클래스 1', '클래스 2', '클래스 0 경계', '클래스 1 경계',
            '클래스 2 경계'], loc=(1.01, 0.3))
plt.title("세 개의 일대다 분류기가 만든 결정 경계")
image.save_fig("blobs_Scatter_Crystal_boundary")  
plt.show()

# 2차원 평면의 모든 포인트에 대한 예측 결과
plt.figure(figsize=(14, 8))
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['클래스 0', '클래스 1', '클래스 2', '클래스 0 경계', '클래스 1 경계',
            '클래스 2 경계'], loc=(1.01, 0.3))
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title("세 개의 일대다 분류기가 만든 다중 클래스 결정 경계")
image.save_fig("blobs_Scatter_Crystal_boundary")  
plt.show()

# 선형 모델의 주요 매개변수는 회귀 모델에서는 alpha,  LinearSVC와 LogisticRegression에서는 C 를 사용
# 중요한 특성이 많지 않다고 생각하면 L1 규제를 사용, 기본적으로 L2 규제를 사용
# L1 규제는 몇 가지 특성만 사용하므로 해당 모델에 중요한 특성이 무엇이고 그 효과가 어느 정도인지 설명하기 용이
# 선형 모델은 학습 속도가 빠르고 예측도 빠릅니다. 
# 매우 큰 데이터셋과 희소한 데이터셋에도 잘 작동합니다. 
# 수십만에서 수백만 개의 샘플로 이뤄진 대용량 데이터셋이라면 기본 설정보다 빨리 처리하도록 
# LogisticRegression과 Ridge에 solver=’sag’ 옵션 20 을 줍니다. 
# 다른 대안으로는 여기서 설명한 선형 모델의 대용량 처리 버전으로 구현된 
# SGDClassifier 와 SGDRegressor21 를 사용


# 메서드 연결
# 한 줄에서 모델의 객체 생성과 학습을 한 번에 실행합니다.
# logreg = LogisticRegression().fit(X_train, y_train)
# y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test)



