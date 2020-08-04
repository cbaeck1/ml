import pandas as pd
import numpy as np
import mglearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import image

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

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=42)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 2. 선형모델 : 로지스틱, 서포트 벡터 머신 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print("2. 선형모델 : 로지스틱 훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("2. 선형모델 : 로지스틱 테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("2. 선형모델 : 로지스틱C=100 훈련 세트 점수: {:.3f}".format(logreg100.score(X_train, y_train)))
print("2. 선형모델 : 로지스틱C=100 테스트 세트 점수: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("2. 선형모델 : 로지스틱C=0.01 훈련 세트 점수: {:.3f}".format(logreg001.score(X_train, y_train)))
print("2. 선형모델 : 로지스틱C=0.01 테스트 세트 점수: {:.3f}".format(logreg001.score(X_test, y_test)))

# 
plt.figure(figsize=(14, 8))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-2, 2)
plt.xlabel("특성")
plt.ylabel("계수 크기")
plt.legend()
plt.title('유방암 데이터셋에 각기 다른 C 값을 사용하여 만든 로지스틱 회귀의 계수')
image.save_fig("cancer_logistic_C")  
plt.show()

# LogisticRegression은 기본으로 L2 규제를 적용 : Ridge로 만든 모습과 비슷
# 세 번째 계수(mean perimeter) 
#   C=100, C=1일 때 이 계수는 음수지만, C=0.001일 때는 양수가 되며 C=1일 때보다도 절댓값이 더 큽니다
# texture error특성은 악성인 샘플과 관련이 깊습니다


# L1 규제(몇 개의 특성만 사용)를 사용할 때의 분류 정확도와 계수 그래프 
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
plt.figure(figsize=(14, 8))
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    # lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    lr_l1 = LogisticRegression(C=C, penalty="none").fit(X_train, y_train)
    print("C={:.3f}인 l1 로지스틱 회귀의 훈련 정확도: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("C={:.3f}인 l1 로지스틱 회귀의 테스트 정확도: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("특성")
plt.ylabel("계수 크기")
plt.ylim(-2, 2)
plt.legend(loc=3)
plt.title('유방암 데이터와 L1 규제를 사용하여 각기 다른 C 값을 적용한 로지스틱 회귀 모델의 계수')
image.save_fig("cancer_logistic_C_L1")  
plt.show()

# 모델들의 주요 차이는 규제에서 모든 특성을 이용할지 일부 특성만을 사용할지 결정하는 
# penalty 매개변수의 갯수이다


# 다중 클래스 분류용 선형 모델