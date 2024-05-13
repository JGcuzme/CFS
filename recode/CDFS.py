import numpy as np
import pandas as pd
import scipy


class CDFS(object):

    def __init__(self, data, CG, CS, labelname):
        """
        data: data set
        CG: cause and effect diagram
        CS: causal strength
        data: 数据集
        CG：因果图
        CS：因果强度
        """
        self.data = data.drop(labelname, axis=1)
        self.CG = CG
        self.CS = CS
        self.label = data[labelname]

        if type(self.data) != "numpy.ndarray":
            self.data = np.array(self.data)
        if type(self.CG) != "numpy.ndarray":
            self.CG = np.array(self.CG)
        if type(self.CS) != "numpy.ndarray":
            self.CS = np.array(self.CS)
        if self.data.shape[1] != self.CG.shape[0] or \
                self.data.shape[1] != self.CS.shape[0] or \
                self.CG.shape[0] != self.CS.shape[0]:
            print("数据维度不相等")
            exit(1)
        # 归一化
        # min_vals = np.min(self.CS)
        # max_vals = np.max(self.CS)
        # self.CS = (self.CS - min_vals) / (max_vals - min_vals)

    def SFBS(self):
        """
        Standard Forward Backward Selection (SFBS)
        标准向前向后选择（SFBS）
        """
        output = []
        infogain = []
        score = 0
        causal_strength = 0
        count = 0
        for i in range(self.data.shape[1]):
            infogain.append(self.infoGain(i))
        # print(infogain)

        # 排序
        index_list = [i[0] for i in sorted(enumerate(infogain), reverse=True, key=lambda x: x[1])]
        print(index_list)

        # 向前搜索
        # 基于因果关系、因果强度和信息增益
        for i in range(self.data.shape[1]):
            if i == 0:
                output.append(index_list[i])
                score += infogain[index_list[i]]
            else:
                for j in range(len(output)):
                    if self.CG[output[j], index_list[i]] == 1 or self.CG[index_list[i], output[j]] == 1:
                        count += 1
                        causal_strength = self.CS[output[j], index_list[i]] + self.CS[index_list[i], output[j]]
                if count != 0:
                    causal_strength = causal_strength / count
                    count = 0
                if score + infogain[index_list[i]] - causal_strength > score:
                    output.append(index_list[i])
                    score = score + infogain[index_list[i]] - causal_strength
                    causal_strength = 0
                else:
                    causal_strength = 0
        # print(output)
        # 向后删除
        # 相关系数
        forward = self.data[:, output]
        forward = pd.DataFrame(forward)
        cor = np.zeros((forward.shape[1], forward.shape[1]))
        p_val = np.zeros((forward.shape[1], forward.shape[1]))
        del_list = []
        for i in range(forward.shape[1]):
            for j in range(forward.shape[1]):
                cor[(i, j)], p_val[(i, j)] = scipy.stats.spearmanr(forward.iloc[:, i], forward.iloc[:, j])
        # print(cor, p_val)
        # np.savetxt(fname="C:/Users/14903/Desktop/cor.csv", X=cor, delimiter=",")
        # np.savetxt(fname="C:/Users/14903/Desktop/p_val.csv", X=p_val, delimiter=",")
        for i in range(forward.shape[1]):
            for j in range(i+1, forward.shape[1]):
                if cor[i, j] >= 0.7 and p_val[i, j] < 0.05:
                    # 删除信息增益较小的的相关变量
                    del_list.append(j)
        # print(del_list)
        del_list = sorted(list(set(del_list)))
        # print(del_list)
        for item in del_list[::-1]:
            del output[item]
        print(output)
        output = self.data[:, output]
        return output

    def IFBS(self):
        """
        Iterative forward backward selection (IFBS)
        交替向前向后选择（IFBS）
        """
        output = []
        infogain = []
        score = 0
        causal_strength = 0
        count = 0
        for i in range(self.data.shape[1]):
            infogain.append(self.infoGain(i))
        # print(infogain)

        # 排序
        index_list = [i[0] for i in sorted(enumerate(infogain), reverse=True, key=lambda x: x[1])]
        print(index_list)

        for i in range(self.data.shape[1]):
            # 添加第一个元素
            if i == 0:
                output.append(index_list[i])
                score += infogain[index_list[i]]
            else:
                for j in range(len(output)):
                    if self.CG[output[j], index_list[i]] == 1 or self.CG[index_list[i], output[j]] == 1:
                        count += 1
                        causal_strength = self.CS[output[j], index_list[i]] + self.CS[index_list[i], output[j]]
                if count != 0:
                    causal_strength = causal_strength / count
                    count = 0
                if score + infogain[index_list[i]] - causal_strength > score:
                    # 向前添加
                    # 因果关系、因果强度和信息增益
                    output.append(index_list[i])
                    score = score + infogain[index_list[i]] - causal_strength
                    causal_strength = 0

                    # 向后删除
                    # 相关系数
                    forward = self.data[:, output]
                    forward = pd.DataFrame(forward)
                    cor = np.zeros((forward.shape[1], forward.shape[1]))
                    p_val = np.zeros((forward.shape[1], forward.shape[1]))
                    del_list = []
                    for i in range(forward.shape[1]):
                        for j in range(forward.shape[1]):
                            cor[(i, j)], p_val[(i, j)] = scipy.stats.spearmanr(forward.iloc[:, i], forward.iloc[:, j])

                    for i in range(forward.shape[1]):
                        for j in range(i + 1, forward.shape[1]):
                            if cor[i, j] >= 0.7 and p_val[i, j] < 0.05:
                                # 删除信息增益较小的的相关变量
                                del_list.append(j)

                    del_list = sorted(list(set(del_list)))

                    for item in del_list[::-1]:
                        del output[item]
                else:
                    causal_strength = 0

        print(output)
        output = self.data[:, output]
        return output

    def infoEntropy(self, label):
        """
        Calculate information entropy
        return: information entropy, type float
        计算信息熵
        return: 信息熵，类型float
        """
        label_set = set(label)
        result = 0
        for l in label_set:
            count = 0
            for j in range(len(label)):
                if label[j] == l:
                    count += 1
            p = count / len(label)
            result -= p * np.log2(p)
        return result

    def HDA(self, index, value):
        """
        Calculate conditional entropy
        return: conditional entropy, type float
        计算条件熵
        return： 条件熵，类型float
        """
        count = 0
        sub_feature = []
        sub_label = []
        for i in range(len(self.data)):
            if self.data[i][index] == value:
                count += 1
                sub_feature.append(self.data[i])
                sub_label.append(self.label[i])
        pHA = count / len(self.data)
        e = self.infoEntropy(sub_label)
        return pHA * e

    def infoGain(self, index):
        """
        Calculate information gain
        return: information gain, type float
        计算信息增益
        return： 信息增益，类型float
        """
        # index -= 1    # 实验用
        base_e = self.infoEntropy(self.label)
        f = np.array(self.data)
        f_set = set(f[:, index])
        sum_HDA = 0
        for value in f_set:
            sum_HDA += self.HDA(index, value)
        return base_e - sum_HDA


if __name__ == '__main__':
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    CG = pd.read_csv(r"C:\Users\14903\Desktop\保存\研究生毕业论文\大论文\实验结果\全部特征（标准化）.txt", sep=",",
                     header=None)
    CS = pd.read_csv(r"C:\Users\14903\Desktop\保存\研究生毕业论文\大论文\实验结果\全部特征因果强度（标准化）.txt",
                     sep=",", header=None)
    # print(data)
    cdfs = CDFS(data=data, CG=CG, CS=CS, labelname="DIQ010")
    feature1 = cdfs.SFBS()
    feature2 = cdfs.IFBS()
