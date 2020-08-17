import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

# 1. 붓꽃iris 데이터셋
# 2. 두 개의 특성을 가진 forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋
# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋
# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격 Boston Housing 데이터셋
# 5. 선형 회귀(최소제곱법)을 위한 wave 데이터셋.  n_samples = 40
# 6. 세 개의 클래스를 가진 간단한 blobs 데이터셋
# 7. scikit-learn에 구현된 나이브 베이즈 분류기는 GaussianNB, BernoulliNB, MultinomialNB 세가지
# 8. 메모리 가격 동향 데이터 셋 ram_prices
# 9. 두 개의 클래스를 가진 2차원 데이터셋 two_moons
# 10. 두 개의 클래스를 가진 2차원 데이터셋
# 11. 
# 12. 뉴스그룹 데이터
# 13. 동물트리

# 1. 히스토그램
# 2. 산점도

# 1. k-최근접 이웃 알고리즘 : 분류, 회귀
#    n_neighbors 변화에 따른 결정 경계   
#    n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
# 2. 선형모델 : 최소제곱, 릿지, 라쏘, 선형분류모델(로지스틱, 서포트벡터머신), 다중분류 선형모델
#    특성공학, 학습곡선    
# 3. 나이브 베이즈 분류기
# 4. 결정트리
#    결정 트리 분석
# 5. 결정트리 앙상블 : 랜덤 포레스트, 그래디언트 부스팅
# 6. 커널서포트벡터머신
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
print(cancer['DESCR']+ "\n...")
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
print(boston['DESCR']+ "\n...")
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])

# 특성 공학feature engineering : load_extended_boston
# 13개의 원래 특성에 13개에서 2개씩 (중복을 포함해) 짝지은 91개의 특성을 더해 총 104개가 됩니다.
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)
print(X[:, 0], y)

# 5. 선형 회귀(최소제곱법)을 위한 wave  데이터셋. n_samples = 40
X, y = mglearn.datasets.make_wave(n_samples=40)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X[:5], y[:5])

# 6. 세 개의 클래스를 가진 간단한 blobs 데이터셋
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
print(boston['DESCR']+ "\n...")
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

# 9. 두 개의 클래스를 가진 2차원 데이터셋 make_moons
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

# 12. 뉴스그룹 데이터
from sklearn.datasets import fetch_20newsgroups
newsdata = fetch_20newsgroups(subset='train')
print(newsdata['DESCR']+ "\n...")
print("cancer.keys(): \n{}".format(newsdata.keys()))
print("뉴스그룹 데이터의 형태: {}".format(newsdata.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(newsdata.target_names, np.bincount(newsdata.target))}))
print("특성 이름:\n{}".format(newsdata.feature_names))
print(newsdata.data, newsdata.target)
print(newsdata.data[:,:2])

