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
# 7. 신경망
# ŷ = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] = b
# ŷ 은 x[0]에서 x[p]까지의 입력 특성과 w[0]에서 w[p]까지 학습된 계수의 가중치 합

# 입력 특성과 예측은 노드node로, 계수는 노드 사이의 연결로 나타낸 로지스틱 회귀
lr = mglearn.plots.plot_logistic_regression_graph()
image.save_graph_as_svg(lr, "9.logistic_regression_plot")  

# 중간 단계를 구성하는 은닉 유닛hidden unit을 계산하고 이를 이용하여 최종 결과를 산출
# 은닉층이 하나인 다층 퍼셉트론
shl = mglearn.plots.plot_single_hidden_layer_graph()
image.save_graph_as_svg(shl, "9.single_hidden_layer_plot")  

# 비선형 함수 : 렐루 rectified linear unit, ReLU 나 하이퍼볼릭 탄젠트 hyperbolic tangent, tanh 를 적용
line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.title("하이퍼볼릭 탄젠트 활성화 함수와 렐루 활성화 함수")
image.save_fig("9.ReLU_tanh_plot")  
plt.show()

# 은닉층으로 구성된 대규모의 신경망이 생기면서 이를 딥러닝
# 은닉층이 두 개인 다층 퍼셉트론
thl = mglearn.plots.plot_two_hidden_layer_graph()
image.save_graph_as_svg(thl, "9.two_hidden_layer_plot")  

#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 은닉 유닛 100개를 사용
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title("은닉 유닛이 100개인 신경망으로 학습시킨 two_moons 데이터셋의 결정 경계")
image.save_fig("9.make_moons_two_moons_hidden100_scatter")  
plt.show()

# 은닉 유닛 10개를 사용
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title("은닉 유닛이 10개인 신경망으로 학습시킨 two_moons 데이터셋의 결정 경계")
image.save_fig("9.make_moons_two_moons_hidden10_scatter")  
plt.show()

# 10개의 유닛으로 된 두 개의 은닉층
mlp = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title("10개의 은닉 유닛을 가진 두 개의 은닉층과 렐루 활성화 함수로 만든 결정 경계")
image.save_fig("9.make_moons_two_hidden_layer_hidden10_scatter")  
plt.show()

#
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
                            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                            alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
plt.title("은닉 유닛과 alpha 매개변수에 따라 변하는 결정 경계")
image.save_fig("9.make_moons_hidden_alpha_scatter")  
plt.show()

# 
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
plt.title("무작위로 다른 초깃값을 주되 같은 매개변수로 학습한 결정 경계")
image.save_fig("9.make_moons_random_hidden_scatter")  
plt.show()



