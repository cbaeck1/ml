import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# iris_dataset의 키: dict_keys(['target_names', 'feature_names', 'DESCR', 'data', 'target'])
print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))

'''
Iris Plants Database
====================
'''

print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

print("target의 타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))


# 성과 측정: 훈련 데이터와 테스트 데이터
# 레이블된 데이터(150개의 붓꽃 데이터)를 두 그룹으로
# 75% 를 레이블 데이터와 함께 훈련 세트로 뽑습니다. 나머지 25%는 레이블 데이터와 함께 테스트 세트
# 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달
# train_test_split 함수의 반환값은 X_train, X_test, y_train, y_test이며 모두 NumPy 배열

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))


# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names 에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe.plot.hist(alpha=0.5)
plt.title("Iris Histogram Plot")
image.save_fig("Iris_Histogram")
plt.show()

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
  hist_kwds={'bins': 20}, s=20, alpha=.8, cmap=mglearn.cm3)
plt.title("Iris Scatter Plot")
image.save_fig("Iris_Scatter")  
plt.show()

# k-최근접 이웃 알고리즘
# scikit-learn의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 각각 구현
# k-최근접 이웃 분류 알고리즘은 neighbors 모듈 아래 KNeighborsClassifier 클래스에 구현
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# knn 객체는 훈련 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화한 것입니다. 
# 또한 알고리즘이 훈련 데이터로부터 추출한 정보를 담고 있습니다.
# KNeighborsClassifier의 경우는 훈련 데이터 자체를 저장하고 있습니다

# 훈련 데이터인 NumPy 배열 X_train과 훈련 데이터의 레이블을 담고 있는 NumPy 배열 y_train을 매개변수
knn.fit(X_train, y_train)

# 예측하기
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(iris_dataset['target_names'][prediction]))

# 모델 평가하기
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n {}".format(y_pred))
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))


