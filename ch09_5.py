import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image


# 9. 두 개의 클래스를 가진 2차원 데이터셋 make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

########################################################################
# 5. 결정트리 앙상블 : 랜덤 포레스트
# 결정 트리의 주요 단점은 훈련 데이터에 과대적합되는 경향
# 랜덤 포레스트는 이 문제를 회피할 수 있는 방법
# 서로 다른 방향으로 과대적합된 트리를 많이 만들어 평균하여 과대적합을 줄이는 방법

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 각 트리가 고유하게 만들어지도록 무작위한 선택
# 데이터의 부트스트랩 샘플bootstrap sample을 생성
# 무작위로 데이터를 n_samples 횟수만큼 반복 추출, 중복 허용
# 특성을 고르는 것은 매 노드마다 반복되므로 트리의 각 노드는 다른 특성들을 사용
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

# max_features 값을 크게 하면 랜덤 포레스트의 트리들은 매우 비슷해지고 가장 두드러진 특성을 이용해 데이터에 잘 맞추고
# max_features 값을 낮추면 랜덤 포레스트 트리들은 많이 달라지고 각 트리는 데이터에 맞추기 위해 깊이가 깊어지게 됩니다.

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
print(type(axes), type(axes.ravel()))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    print(type(ax))
    ax.set_title("트리 {}".format(i))
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("랜덤 포레스트")
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

image.save_fig("9.make_moons_tree_random_forest")  
plt.show()
