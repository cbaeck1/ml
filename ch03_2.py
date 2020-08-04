import pandas as pd
import numpy as np
import mglearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import image


# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격Boston Housing 데이터셋
# 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 1970년대 보스턴 주변의 주택 평균 가격을 예측
# 이 데이터셋에는 데이터 포인트 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print("data의 타입: {}".format(type(boston['data'])))
print("data의 크기: {}".format(boston['data'].shape))
print("data의 처음 다섯 행:\n{}".format(boston['data'][:5]))

print("타깃의 이름: {}".format(boston['target']))
print("target의 타입: {}".format(type(boston['target'])))
print("target의 크기: {}".format(boston['target'].shape))
print("타깃:\n{}".format(boston['target']))

# 특성 공학feature engineering : load_extended_boston
# 13개의 원래 특성에 13개에서 2개씩 (중복을 포함해) 짝지은 91개의 특성을 더해 총 104개가 됩니다.
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)
print(X[:, 0], y)

# 산점도를 그립니다. 1개 특성, 1개 타겟
plt.plot(X[:, 0], y, 'x')
plt.ylim(0, 60)
plt.xlabel("특성")
plt.ylabel("타겟")
plt.title("Boston Scatter Plot")
image.save_fig("Boston_Scatter")  
plt.show()

# 훈련 세트, 테스트 세트 random_state=0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
















