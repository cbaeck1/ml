import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image
from IPython.display import display 
import graphviz

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
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 4. 결정트리
# 유방암 데이터셋을 이용하여 사전 가지치기의 효과를 확인
from sklearn.tree import DecisionTreeClassifier
tree0 = DecisionTreeClassifier(random_state=0)
tree0.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree0.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree0.score(X_test, y_test)))

# max_depth=4 
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# 결정 트리 분석
# export_graphviz 함수를 이용해 트리를 시각화
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="./images/svg/tree.dot", class_names=["악성", "양성"],
                feature_names=cancer.feature_names,
                impurity=False, filled=True)
with open("./images/svg/tree.dot", encoding='utf8') as f:
    dot_graph = f.read()

# display(graphviz.Source(dot_graph))
# print(graphviz.Source(dot_graph))
image.save_graph_as_svg(dot_graph, "cancer_decision_tree4")  

# 트리의 특성 중요도 (feature importance)
# 각 특성에 대해 0은 전혀 사용되지 않았다는 뜻이고 1은 완벽하게 타깃 클래스를 예측했다는 의미
print("특성 중요도:\n{}".format(tree.feature_importances_))

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
plt.title('유방암 데이터로 학습시킨 결정 트리의 특성 중요도')
image.save_fig("cancer_tree_feature_importances")  
plt.show()


# 특성과 클래스 사이에는 간단하지 않은 관계가 있음에 관한 예제
# X[1]에 있는 정보만 사용되었고 X[0]은 전혀 사용되지 않음
# 
tree_not_monotone = mglearn.plots.plot_tree_not_monotone()
plt.title('데이터로 학습한 결정 트리')
image.save_fig("cancer_tree_not_monotone")  
plt.show()



