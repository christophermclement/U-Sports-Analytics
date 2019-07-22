# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:24:50 2018

@author: Chris Clement
"""

import random
import numpy
import Globals
import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

start = datetime.datetime.now().timestamp()


def timestamp():
    '''
    Returns a timestamp in seconds elapsed since the program started running
    '''
    return ("{0:4.2f}".format(datetime.datetime.now().timestamp() - start))


def Binom(N, P, A, B):
    '''
    This is used by BinomHigh and BinomLow to calculate the iterative steps to find the CI
    '''
    Q = P / (1 - P)
    K = 0
    V = 1
    S = 0
    T = 0
    while K <= N:
        T = T + V
        if K >= A and K <= B:
            S = S + V
        if T > 10 ** 30:
            S = S / 10 ** 30
            T = T / 10 ** 30
            V = V / 10 ** 30
        K = K + 1
        V = V * Q * (N + 1 - K) / K
    return S/T


def BinomLow(X, N, C):
    '''
    Given a confidence value it returns the lowerCI of a binomial distribution
    X - number of successes
    N - number of failures
    C - Confidence value
    This could stand to be made more robust by checking the inputs
    '''
    P = X / N  # Probability of success
    V = P / 2  # Half of probability
    L = numpy.float64(0.0)
    H = P
    while (H - L) > 10 ** (-12):  # TODO: Crank up the precision if we want
        if Binom(N, V, X, N) > C:
            H = V
            V = (L + V) / 2
        else:
            L = V
            V = (V + H) / 2
    return V


def BinomHigh(X, N, C):
    '''
    Given a confidence value it returns the upper CI of a binomial distribution
    X - number of successes
    N - number of failures
    C - Confidence value
    This could stand to be made more robust by checking the inputs
    '''
    P = X / N
    V = (1 + P) / 2
    L = P
    H = numpy.float64(1.0)
    while (H - L) > 10 ** (-12):
        if Binom(N, V, 0, X) < C:
            H = V
            V = (L + V) / 2
        else:
            L = V
            V = (V + H) / 2
    return V


def bootstrap(values, C):
    '''
    This creates a bootstrap array, where it creates bootstrap samples. We no longer need this function, we made numpy do it
    '''
    straplist = numpy.empty(Globals.BOOTSTRAP_SIZE)
    for y in range(Globals.BOOTSTRAP_SIZE):
        Tot = 0
        for x in range(len(values)):
            Tot = Tot + random.choice(values)
        straplist[y] = (Tot / len(values))
    straplist.sort()
    straplist = numpy.array(straplist, float)
    CLow = straplist[int(C * Globals.BOOTSTRAP_SIZE - 1)]
    CHigh = straplist[(1 - C) * Globals.BOOTSTRAP_SIZE]
    return [CLow, CHigh, straplist]
 

def BootCompare(arrA, arrB):
    count = b = 0
    for a in range(Globals.BOOTSTRAP_SIZE):
        for b in range(b, Globals.BOOTSTRAP_SIZE):
            if arrB[b] > arrA[a]:
                count += b
                break
        else:
            count += Globals.BOOTSTRAP_SIZE
    count /= Globals.BOOTSTRAP_SIZE**2
    return numpy.float64(count)


functions_dict = {"linearFit": a * x + b,
                  "exponentialDecayFit": a * numpy.exp(x * b) + c,
                  "logarithmicFit": a * numpy.log(x + b) + c,
                  "quadraticFit": a * x ** 2 + b * x + c,
                  "reverseQuadraticFit": a * (x + b) ** 0.5 + c,
                  "sigmoidFit": a / (b + numpy.exp(- c * x)),
                  "inverseSigmoidFit": numpy.log(a / x + b) + c,
                  "tangentFit": a * numpy.tan(b + x) + c,
                  "cubicFit": a * x ** 3 + b * x ** 2 + c * x + d}


def fit_functions(model, x, a=0, b=0, c=0, d=0, e=0):
    '''
    Put in a keyword and it give us back the function to go with, but this is a lot cleaner than using a zillion one-liners, now we just reference a dict
    TODO: TBH we could probably just reference the dict directly
    '''
    return(models_dict[model])


def RMSE(func, params, xdata, ydata):
    '''
    Returns the RMSE of a set of xy data vs a function
    '''
    return numpy.sqrt(((func(xdata, *params) - ydata) ** 2).mean())


def RSquared(func, params, xdata, ydata):
    '''
    Returns the r2 of a set of xy data vs a function
    '''
    residuals = ydata - func(xdata, *params)
    ss_res = numpy.sum(residuals ** 2)
    ss_tot = numpy.sum((ydata - numpy.mean(ydata)) ** 2)
    return (1 - (ss_res / ss_tot))


def ordinals(num):
    '''
    This returns the ordinal string for any integer given
    '''
    try:
        if int(num) != num:
            print("not an int!", num)
        if len(str(num)) > 1:
            if str(num)[-2:-1] == "1":
                return str(num) + "$^{th}$"
        elif str(num)[-1] == "1":
            return str(num) + "$^{st}$"
        elif str(num)[-1] == "2":
            return str(num) + "$^{nd}$"
        elif str(num)[-1] == "3":
            return str(num) + "$^{rd}$"
        else:
            return str(num) + "$^{th}$"
    except Exception as err:
        print("Ordinal Error", num)
        print(err)
    return "ordinals error, wtf?"


labels_dict = {"linearFit": r"$y={0:5.4f}*x+{1:5.4f}$" + "\n" + "$R^2={2:5.4f}, RMSE={3:5.4f}$",
               "quadraticFit": r"$y={0:5.4f}*x^2+{1:5.4f}*x+{2:5.4f}$" + "\n" + r"$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "exponentialDecayFit": r"$y={0:5.4f}*e^({1:5.4f}*x)+{2:5.4f}$" + "\n" + r"$R^2={3:5.4f}$, $RMSE={4:5.4f}$",
               "logarithmicFit": r"$y={0:5.4f}*ln({1:5.4f}+x)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "reverseQuadraticFit": r"$y={0:5.4f}*(x+{1:5.4f}^(0.5)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "sigmoidFit": r"$y={0:5.4f}/({1:5.4f} + e^(-{2:5.4f}*x))$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "tangentFit": r"$y={0:5.4f}*tan({1:5.4f}+x)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "inverseSigmoidFit": r"$y=ln({0:5.4f}/x+{1:5.4f})+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$",
               "cubicFit": r"$y={0:5.4f}*x^3+{1:5.4f}*x^2+{2:5.4f}*x+{3:5.4f}$" + "\n" + "$R^2={4:5.4f}, RMSE={5:5.4f}$"}


def fitLabels(func):
    '''
    Returns a formatted string to give the label of a function from those used above, with R2 and RMSE.
    References a dict to be much easier to look up
    TODO: TBH we could probably just call the dict directly
    '''
    return labels_dict[func]


def imscatter(x, y, image, ax=None, zoom=1):
    '''
    Use this to create a graph with logos on it based on images
    '''
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = numpy.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(numpy.column_stack([x, y]))
    ax.autoscale()
    return artists

def printFeatures(modellist):
    for model in modellist:
        if hasattr(model, "coef_"):
            print(type(model).__name__, "coef_")
            print(model.coef_)
        if hasattr(model, "intercept_"):
            print(type(model).__name__, "intercept_")
            print(model.intercept_)            
        if hasattr(model, "feature_importances_"):
            print(type(model).__name__, "feature_importances")
            print(model.feature_importances_)            
    return None