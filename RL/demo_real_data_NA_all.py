import numpy as np

from recode.utils import output_adj
from recode.rl import RL
import pandas as pd
from recode.causal_strength import adj_cs

# X = pd.read_csv(r"../data/dataset/all.csv")  # 全部数据无标准化
# X = pd.read_csv(r"../data/dataset/all_standard.csv")  # 全部数据标准化

X = pd.read_csv(r"../data/dataset/LMCH.csv")  # LMCH无标准化
# X = pd.read_csv(r"../data/dataset/LMCH_standard.csv")  # LMCH标准化
# X = np.array(X.drop(columns="DIQ010"))
X = np.array(X.drop(columns="CLASS"))

rl = RL(nb_epoch=10000, device_type="cpu", score_type="BIC")
rl.learn(X)
print(rl.causal_matrix)  # 因果矩阵
print(adj_cs(rl.causal_matrix, X))  # 因果强度
np.savetxt(r"../data/dataset/LMCH.txt", rl.causal_matrix, fmt="%d", delimiter=", ")
np.savetxt(r"../data/dataset/LMCH因果强度.txt", adj_cs(rl.causal_matrix, X), fmt="%.4f", delimiter=", ")

output_adj(rl.causal_matrix)  # 因果图

