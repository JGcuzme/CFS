from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from mrmr import mrmr_classif
from tqdm import tqdm
from sklearn.decomposition import PCA
from recode import CDFS
from recode import relief
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import pandas as pd
import time
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
pd.set_option("max_colwidth", 200)
import warnings
warnings.filterwarnings("ignore")


def feature_selection_rfe_svr(data, feature_num):
    """使用递归特征消除进行特征选择"""
    y = data.iloc[:, -1]
    X = data.drop(data.columns[-1], axis=1)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=feature_num, step=1)
    selector = selector.fit(X, y)
    selected_data = X.iloc[:, selector.get_support(indices=True)]
    return selected_data


def feature_selection_mrmr(data, feature_num):
    """使用mRMR进行特征选择"""
    y = data.iloc[:, -1]
    X = data.drop(data.columns[-1], axis=1)
    selected_data = mrmr_classif(X=X, y=y, K=feature_num)
    selected_data = X[selected_data]
    return selected_data


def feature_selection_rrelieff(data, feature_num):
    """使用 RReliefF进行特征选择"""
    y = data.iloc[:, -1]
    X = data.drop(data.columns[-1], axis=1)
    w = relief.RReliefF(X.values, y.values)
    sorted_id = sorted(range(len(w)), key=lambda k: w[k], reverse=True)[:feature_num]
    selected_data = X.iloc[:, sorted_id]
    return selected_data


def feature_selection_pca(data, feature_num):
    """使用 pca进行特征选择"""
    y = data.iloc[:, -1]
    X = data.drop(data.columns[-1], axis=1)
    pca = PCA(n_components=feature_num)
    # 对数据集进行拟合和转换
    selected_data = pca.fit_transform(X)
    return selected_data


def RFC_1():
    """RF: NHANES data"""
    dataset = pd.read_csv(r"../data/dataset/all_standard.csv")
    target = dataset["DIQ010"]
    dataset = dataset.drop(columns="DIQ010")

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


def RFC_2():

    """RF: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    CG = pd.read_csv(r"../data/files/NA_CG.txt", sep=",", header=None)
    CS = pd.read_csv(r"../data/files/NA_CS.txt", sep=",", header=None)

    cdfs = CDFS.CDFS(data=data, CG=CG, CS=CS, labelname="DIQ010")
    feature = cdfs.SFBS()
    target = data["DIQ010"]
    dataset = feature

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


def RFC_3():

    """RF: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    feature = feature_selection_rfe_svr(data, 30)
    target = data["DIQ010"]
    dataset = feature

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


def RFC_4():

    """RF: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    feature = feature_selection_mrmr(data, 30)
    target = data["DIQ010"]
    dataset = feature

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


def RFC_5():

    """RF: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    feature = feature_selection_rrelieff(data, 30)
    target = data["DIQ010"]
    dataset = feature

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


def RFC_6():

    """RF: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    feature = feature_selection_pca(data, 30)
    target = data["DIQ010"]
    dataset = feature

    test_precision = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    etime = 0

    for i in tqdm(range(1000), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)
        start = time.perf_counter()
        ABC1 = RandomForestClassifier(oob_score=True, random_state=2, max_samples=64)
        ABC1.fit(X_train, Y_train)
        ABC1_test = ABC1.predict(X_test)

        end = time.perf_counter()

        etime = etime + (end - start)
        test_precision = test_precision + accuracy_score(Y_test, ABC1_test)
        precision = precision + precision_score(Y_test, ABC1_test, average='macro')
        recall = recall + recall_score(Y_test, ABC1_test, average='macro')
        f1 = f1 + f1_score(Y_test, ABC1_test, average='macro')
        pre_test = ABC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)

    etime = etime / 1000
    AUC = AUC / 1000
    print("准确率：", test_precision / 1000)
    print("精确率：", precision / 1000)
    print("召回率：", recall / 1000)
    print("F1值：", f1 / 1000)
    print('程序运行时间为: %s 秒' % etime)
    print("AUC: %s" % AUC)


if __name__ == '__main__':


    RFC_2()
    # RFC_3()
    # RFC_4()
    # RFC_5()
    # RFC_6()
