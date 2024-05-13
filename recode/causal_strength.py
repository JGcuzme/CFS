import numpy as np
import math
from scipy.special import psi


def adj_cs(adj_matrix, data):
    """
    Calculate the causal strength of each causal edge
    :param adj_matrix: causal adjacency matrix
    :param data: data
    :return: causal strength matrix
    """
    num = 0
    if type(data) != "numpy.ndarray":
        data = np.asarray(data)
    dim = adj_matrix.shape[0]
    strength_matrix = np.array(np.zeros((dim, dim)))
    for i in range(dim):
        for j in range(dim):
            if adj_matrix[i, j] == 1:
                strength_matrix[i, j] = causal_strength(data[:, i], data[:, j])
                num = num + 1
    # print(num)
    # 矩阵归一化
    min_vals = np.min(strength_matrix)
    max_vals = np.max(strength_matrix)
    strength_matrix = (strength_matrix - min_vals) / (max_vals - min_vals)
    return strength_matrix


def causal_strength(sam1, sam2, refMeasure=2):
    """
    Computes the causal strength between two variables
    :param sam1: sample 1
    :param sam2: sample 2
    :param refMeasure: Normalization method
    :return: causal strength(float)
    """
    # Take the real part
    sam1 = np.real(sam1)
    sam2 = np.real(sam2)
    # Check the entered parameters
    len1 = len(sam1)
    len2 = len(sam2)
    if len1 < 20:
        print("Not enough observations in sam1 (must be > 20)")
        exit(1)
    if len2 < 20:
        print("Not enough observations in sam2 (must be > 20)")
        exit(1)
    if len1 != len2:
        print("Lenghts of sam1 and sam2 must be equal")
        exit(1)
    # Normalization and standardization
    if refMeasure == 1:  # Normalization
        sam1 = (sam1 - min(sam1)) / (max(sam1) - min(sam1))
        sam2 = (sam2 - min(sam2)) / (max(sam2) - min(sam2))
    if refMeasure == 2:   # standardization
        sam1 = (sam1 - np.mean(sam1)) / np.std(sam1)
        sam2 = (sam2 - np.mean(sam2)) / np.std(sam2)
    if refMeasure != 1 and refMeasure != 2:
        print("Warning: unknown reference measure - no scaling applied")
        exit(1)
    # entropy estimation
    ind1 = np.sort(sam1)
    ind2 = np.sort(sam2)

    # Entropy estimate for sample 1
    hx = 0
    for i in range(len1 - 1):
        delta = ind1[i + 1] - ind1[i]
        if delta != 0:
            hx = hx + math.log(abs(delta))
    hx = hx / (len1 - 1) + psi(len1) - psi(1)

    # Entropy estimate for sample 2
    hy = 0
    for i in range(len2 - 1):
        delta = ind2[i + 1] - ind2[i]
        if delta != 0:
            hy = hy + math.log(abs(delta))
    hy = hy / (len1 - 1) + psi(len2) - psi(1)

    # Calculate causal strength
    strength = 1 / abs(hy - hx)

    return strength


