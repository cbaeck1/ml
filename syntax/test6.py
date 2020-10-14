import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

max_depthList = [1, 10, 100, 1000]
learning_rateList = [0.001, 0.01, 0.1, 1, 10]

# fig란 figure로써 - 전체 subplot을 말한다. ex) 서브플로안에 몇개의 그래프가 있던지 상관없이  그걸 담는 하나. 전체 사이즈를 말한다.
# axes는 axe로써 - 전체 중 낱낱개를 말한다 ex) 서브플롯 안에 2개(a1, a2)의 그래프가 있다면 a1, a2 를 일컬음
fig, axes = plt.subplots(len(max_depthList), len(learning_rateList), figsize=(len(max_depthList)*3, len(learning_rateList)*2))



plt.show()