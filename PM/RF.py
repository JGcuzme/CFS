"""
Standard feature selection function: SFBS
Iterative feature selection function: IFBS
"""

import time
import matplotlib
from recode import CDFS
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from tqdm import tqdm


def RFC1():
    """Random Forest: PIMA data"""
    dataset = pd.read_csv(r"../data/dataset/768_1.csv")
    target = dataset["Outcome"]
    dataset = dataset.drop(columns="Outcome")

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True, random_state=i, max_samples=32)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)


def RFC2():
    """Random Forest: PIMA data"""
    data = pd.read_csv(r"../data/dataset/768_1.csv")
    CG = pd.read_csv(r"../data/files/768_CG.txt", sep=",", header=None)
    CS = pd.read_csv(r"../data/files/768_CS.txt", sep=",", header=None)
    cdfs = CDFS.CDFS(data=data, CG=CG, CS=CS, labelname="Outcome")
    feature = cdfs.IFBS()
    target = data["Outcome"]
    dataset = feature

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True,  random_state=i, max_samples=32)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)


def RFC5():
    """Random Forest: NHNAES data"""
    dataset = pd.read_csv(r"../data/dataset/all_standard.csv")
    target = dataset["DIQ010"]
    dataset = dataset.drop(columns="DIQ010")

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True, random_state=i, max_samples=64)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)


def RFC6():
    """Random Forest: NHANES data"""
    data = pd.read_csv(r"../data/dataset/all_standard.csv")
    CG = pd.read_csv(r"../data/files/NA_CG.txt", sep=",", header=None)
    CS = pd.read_csv(r"../data/files/NA_CS.txt", sep=",", header=None)

    cdfs = CDFS.CDFS(data=data, CG=CG, CS=CS, labelname="DIQ010")
    feature = cdfs.IFBS()
    target = data["DIQ010"]
    dataset = feature

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True,  random_state=i, max_samples=64)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)


def RFC13():
    """Random Forest: LMCH data"""
    dataset = pd.read_csv(r"../data/dataset/LMCH_standard.csv")
    target = dataset["CLASS"]
    dataset = dataset.drop(columns="CLASS")

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True, random_state=i, max_samples=32)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)

def RFC14():
    """Random Forest: LMCH data"""
    data = pd.read_csv(r"../data/dataset/LMCH_standard.csv")
    CG = pd.read_csv(r"../data/files/LMCH_CG.txt", sep=",", header=None)
    CS = pd.read_csv(r"../data/files/LMCH_CS.txt", sep=",", header=None)

    cdfs = CDFS.CDFS(data=data, CG=CG, CS=CS, labelname="CLASS")
    feature = cdfs.IFBS()
    target = data["CLASS"]
    dataset = feature

    epoch = 1000
    etime = 0
    train_precision = 0
    test_precision = 0
    AUC = 0
    for i in tqdm(range(epoch), desc="Training"):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, target,
                                                            test_size=0.3, random_state=i)

        start = time.perf_counter()
        RFC1 = RandomForestClassifier(oob_score=True, random_state=i, max_samples=32)
        RFC1.fit(X_train, Y_train)
        RFC1_train = RFC1.predict(X_train)
        RFC1_test = RFC1.predict(X_test)
        end = time.perf_counter()
        etime = etime + (end - start)
        train_precision += accuracy_score(Y_train, RFC1_train)
        test_precision += accuracy_score(Y_test, RFC1_test)
        pre_test = RFC1.predict_proba(X_test)[:, 1]
        FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
        AUC = AUC + auc(FPR_NB, TPR_NB)
    etime = etime / epoch
    train_precision = train_precision / epoch
    test_precision = test_precision / epoch
    AUC = AUC / epoch
    print('程序运行时间为: %s 秒' % etime)
    print("Precision on the training data set:", train_precision)
    print("Precision on the testing data set:", test_precision)
    print("AUC: %s" % AUC)


if __name__ == '__main__':

    print("---------------------PIMA original features------------------------")
    RFC1()

    print("---------------------PIMA feature selection------------------------")
    RFC2()

    print("--------------------NHANES original features-----------------------")
    RFC5()

    print("--------------------NHANES feature selection-----------------------")
    RFC6()

    print("---------------------LMCH original features------------------------")
    RFC13()

    print("---------------------LMCH feature selection------------------------")
    RFC14()

