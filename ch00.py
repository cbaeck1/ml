import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 1. 붓꽃iris 데이터셋
# 2. 두 개의 특성을 가진 forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋
# 3. 위스콘신 유방암Wisconsin Breast Cancer 데이터셋
# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격Boston Housing 데이터셋
# 5. 세 개의 클래스를 가진 간단한 데이터셋
# 6. 세 개의 클래스를 가진 간단한 데이터셋
# 7. scikit-learn에 구현된 나이브 베이즈 분류기는 GaussianNB, BernoulliNB, MultinomialNB 세가지
# 8. 메모리 가격 동향 데이터 셋
# 9. 두 개의 클래스를 가진 2차원 데이터셋
# 10. 두 개의 클래스를 가진 2차원 데이터셋
# 11. 


# 1. k-최근접 이웃 알고리즘
# 2. 선형모델 : 최소제곱, 릿지, 라쏘, 로지스틱, 다중 클래스 분류용 선형 모델
# 3. 나이브 베이즈 분류기
# 4. 결정트리
# 5. 결정트리 앙상블
# 6. 서포트벡터머신
# 7. 신경망



# 1. 붓꽃iris 데이터셋
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

print("target의 타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))

# 2. 두 개의 특성을 가진 forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋
X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)


# 3. 위스콘신 유방암Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer라고 하겠습니다). 
# 각 종양은 양성benign(해롭지 않은 종양)과 악성malignant(암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격Boston Housing 데이터셋을 사용
# 이 데이터셋으로 할 작업은 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 
# 1970년대 보스턴 주변의 주택 평균 가격을 예측하는 것입니다. 
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])

# 5. 선형 회귀(최소제곱법)을 위한 데이터셋. n_samples = 40
X, y = mglearn.datasets.make_wave(n_samples=40)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 6. 세 개의 클래스를 가진 간단한 데이터셋
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 7. scikit-learn에 구현된 나이브 베이즈 분류기는 GaussianNB, BernoulliNB, MultinomialNB 세가지
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
    # 클래스마다 반복
    # 특성마다 1이 나타난 횟수를 센다.
    counts[label] = X[y == label].sum(axis=0)
print("특성 카운트:\n{}".format(counts))


# 8. 메모리 가격 동향 데이터 셋
import pandas as pd
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
print("ram_prices.keys(): \n{}".format(ram_prices.keys()))
print("데이터의 형태: {}".format(ram_prices.shape))
print(ram_prices[:5])

# 2000년 이전을 훈련 데이터로, 2000년 이후를 테스트 데이터로 만듭니다.
data_train = ram_prices[ram_prices.date < 2000] 
data_test = ram_prices[ram_prices.date >= 2000]
# 가격 예측을 위해 날짜 특성만을 이용합니다.
X_train = data_train.date[:, np.newaxis]
# 데이터와 타깃의 관계를 간단하게 만들기 위해 로그 스케일로 바꿉니다.
y_train = np.log(data_train.price)
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print(X_train[:5], y_train[:5])

# 9. 두 개의 클래스를 가진 2차원 데이터셋
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print(X_train[:5], y_train[:5])

# 10. 두 개의 클래스를 가진 2차원 데이터셋
from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 11. 
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
X, y = mglearn.tools.make_handcrafted_dataset()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

