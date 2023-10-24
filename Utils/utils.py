import numpy as np
import gc
import math
import pyunidoe as uni
from math import gcd as bltin_gcd
from scipy.special import comb, perm
from itertools import combinations, permutations
from random import sample





def glp(n, p, type='CD2'):
    '''
    Good lattice point method, to give a uniform design.
    :param n: number of points
    :param p: dimension
    :param type:  criterion
    :return: points
    '''

    if p == 1:
        design0 = np.zeros((n, p))
        for i in range(n):
            design0[i] = (2 * (i + 1) - 1) / (2 * n)
    elif p == 2 and (n + 1) in np.array((3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)):
        fb = np.array((3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987))
        H = np.array((1, fb[np.where(n <= fb)[0][0]]))
        design = np.zeros((n + 1, p))
        for i in range(p):
            for j in range(n + 1):
                design[j, i] = (2 * (j + 1) * H[i] - 1) / (2 * (n + 1)) - np.floor(
                    (2 * (j + 1) * H[i] - 1) / (2 * (n + 1)))
        design0 = design[0:n, :] * (n + 1) / (n)
    else:
        h = np.array(())
        for i in range(n - 1):
            if coprime2(i + 2, (n + 1)):
                h = np.append(h, i + 2)
        for i in range(100):
            if comb((p + i), i) > 5000:
                addnumber = i
                break
        h0 = h[sample(range(len(h)), min(len(h), (p + addnumber)))]
        H = list(combinations(h0, p))
        if len(H) > 3000:
            H_ = np.array(H)
            H_ = H_[sample(range(len(H)), 3000)]
            H = list(H_)
        design0 = np.ones((n, p))
        d0 = uni.design_eval(design0, type)
        for t in range(len(H)):
            design = np.zeros((n, p))
            for i in range(p):
                for j in range(n):
                    design[j, i] = ((j + 1) * H[t][i]) % (n + 1)
            d1 = uni.design_eval(design, type)
            if d1 < d0:
                d0 = d1
                design0 = design
        design0 = (design0 * 2 - 1) / (2 * n)
        gc.collect()

    return design0

def coprime2(a, b):
    return bltin_gcd(a, b) == 1

def uni_count(point: np.ndarray):

    if len(point.shape) == 0:
        return point, np.array([1])

    elif len(point.shape) == 1:
        point = point.reshape(point.shape[0], 1)

    if point.shape[0] == 1:
        return point, np.array([1])

    point_uni = [point[0, :]]
    count = [1]
    n = point.shape[0]
    for i in range(n - 1):
        temp = point[(i + 1), :]
        for j in range(len(point_uni)):
            if (point_uni[j] == temp).all():
                count[j] += 1
                break

            if j == (len(point_uni) - 1):
                point_uni.append(temp)
                count.append(1)

    return np.array(point_uni), np.array(count)

def eval(y_predict, y_ture, pit = True):
    """
    Eval the percentage of y == yp
    """
    d1 = y_predict.shape[0]
    d2 = y_ture.shape[0]
    assert d1 == d2, "the numbers of two ys don`t match. Please check the number."
    acc = (y_ture == y_predict).sum() / len(y_predict)
    if pit == True:
        print("The accuracy is %.2f%%." % (acc * 100))
    return np.around((acc * 100), 2)

def logit(x):
    """
    Logit function
    """
    ones = np.ones_like(x)
    y = ones / (ones + math.e ** x)
    return y

def classify(y):
    """
    if y > 0.5, then return 1, else 0.
    r is the bool mask.
    """
    r = y > 0.5
    result = r.astype('int')
    return result, r