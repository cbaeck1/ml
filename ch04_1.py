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

# 훈련 세트, 테스트 세트 random_state=66
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 boston.feature_names 에 있는 문자열을 사용합니다.
# 사용할 특성의 갯수을 설정
nCase = 10
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
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
# 훈련 세트를 사용하여 분류 모델을 학습
clf.fit(X_train, y_train)
print("훈련 세트 정확도: {:.2f}".format(clf.score(X_train, y_train)))
# 예측
prediction = clf.predict(X_test)
print("테스트 세트 예측: {}".format(prediction))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))

# n_neighbors 값이 각기 다른 최근접 이웃 모델이 만든 결정 경계
# 이웃의 수를 늘릴수록 결정 경계는 더 부드러워집니다
# 2개 특성만으로 
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend(loc=3)
image.save_fig("Boston_KNN_n_neighbors_1_3_9")  
plt.show()


# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
image.save_fig("Boston_KNN_n_neighbors_1_10")  
plt.show()









