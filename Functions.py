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
from sklearn.model_selection import KFold
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
    L = numpy.float(0.0)
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
    H = numpy.float(1.0)
    while (H - L) > 10 ** (-12):
        if Binom(N, V, 0, X) < C:
            H = V
            V = (L + V) / 2
        else:
            L = V
            V = (V + H) / 2
    return V


def bootstrap(input_list):
    '''
    This creates a bootstrap array, where it creates bootstrap samples. We no longer need this function, we made numpy do it
    '''

    return numpy.sort(numpy.average(numpy.random.choice(
        input_list, (Globals.BOOTSTRAP_SIZE, len(input_list)), replace=True), axis=1))

 
def boot_compare(arrA, arrB):
    count = b = 0
    for a in range(Globals.BOOTSTRAP_SIZE):
        for b in range(b, Globals.BOOTSTRAP_SIZE):
            if arrB[b] > arrA[a]:
                count += b
                break
        else:
            count += Globals.BOOTSTRAP_SIZE
    count /= Globals.BOOTSTRAP_SIZE**2
    return numpy.float(count)


def linearFit(x, a, b):
    return a * x + b
def exponentialDecayFit(x, a, b, c):
    return a * numpy.exp(x * b) + c
def logarithmicFit(x, a, b, c):
    return a * numpy.log(x + b) + c
def quadraticFit(x, a, b, c):
    return a * x ** 2 + b * x + c
def reverseQuadraticFit(x, a, b, c):
   return a * (x + b) ** 0.5 + c
def sigmoidFit(x, a, b, c):
   return a / (b + numpy.exp(- c * x))
def inverseSigmoidFit(x, a, b, c):
   return numpy.log(a / x + b) + c
def tangentFit(x, a, b, c):
   return a * numpy.tan(b + x) + c
def cubicFit(x, a, b, c, d):
   return a * x ** 3 + b * x ** 2 + c * x + d


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


def fitLabels(func):
    '''
    Returns a formatted string to give the label of a function from those used above, with R2 and RMSE
    '''
    if func == linearFit:
        return r"$y={0:5.4f}*x+{1:5.4f}$" + "\n" + "$R^2={2:5.4f}, RMSE={3:5.4f}$"
    elif func == quadraticFit:
        return r"$y={0:5.4f}*x^2+{1:5.4f}*x+{2:5.4f}$" + "\n" + r"$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == exponentialDecayFit:
        return r"$y={0:5.4f}*e^({1:5.4f}*x)+{2:5.4f}$" + "\n" + r"$R^2={3:5.4f}$, $RMSE={4:5.4f}$"
    elif func == logarithmicFit:
        return r"$y={0:5.4f}*ln({1:5.4f}+x)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == reverseQuadraticFit:
        return r"$y={0:5.4f}*(x+{1:5.4f}^(0.5)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == sigmoidFit:
        return r"$y={0:5.4f}/({1:5.4f} + e^(-{2:5.4f}*x))$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == tangentFit:
        return r"$y={0:5.4f}*tan({1:5.4f}+x)+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == inverseSigmoidFit:
        return r"$y=ln({0:5.4f}/x+{1:5.4f})+{2:5.4f}$" + "\n" + "$R^2={3:5.4f}, RMSE={4:5.4f}$"
    elif func == cubicFit:
        return r"$y={0:5.4f}*x^3+{1:5.4f}*x^2+{2:5.4f}*x+{3:5.4f}$" + "\n" + "$R^2={4:5.4f}, RMSE={5:5.4f}$"
    return "fitLabels error" # Catchall escape


def fit_models(model_list, xdata, ydata, returns):
    if returns > 1:
        outputlist = numpy.empty((len(model_list), 0, returns))
    else:
        outputlist = numpy.empty((len(model_list), 0))
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(xdata)
    for train_index, test_index in kf.split(xdata):
        temp = []
        for m, model in enumerate(model_list):
            model.fit(xdata.iloc[train_index], ydata.iloc[train_index].values.ravel())
            if hasattr(model, "predict_proba"):
                temp.append(model.predict_proba(xdata.iloc[test_index]))
            else:
                temp.append(model.predict(xdata.iloc[test_index]))
            print("\t", type(model).__name__, "fitted", timestamp())
        temp = numpy.array(temp)

        outputlist = numpy.concatenate((outputlist, temp), axis=1)
    print("    KFolds fitted", timestamp())

    for model in model_list:
        model.fit(xdata, ydata.values.ravel())
        print("\t", type(model).__name__, "fitted", timestamp())
    print("    Full models fitted", timestamp())
    return outputlist


def correlation_graph(input_data, ax):
    
    corr_data = [[] for x in range(101)]
    for datum in input_data:
        corr_data[int(round(datum[0] * 100))].append(datum[1])
    xdata = []
    err = []
    ydata = []
    for d, datum in enumerate(corr_data):
        if len(datum) > Globals.THRESHOLD:
            ydata.append(numpy.mean(datum) * 100)
            err.append([(ydata[-1] - BinomLow(sum(datum), len(datum), Globals.CONFIDENCE)) * 100,
                        (BinomHigh(sum(datum), len(datum), Globals.CONFIDENCE) - ydata[-1]) * 100])
            xdata.append(d)
    err = numpy.transpose(err)
    xdata = numpy.array(xdata)
    ydata = numpy.array(ydata)
    rmse = RMSE(linearFit, [1, 0], xdata, ydata)
    r2 = RSquared(linearFit, [1, 0], xdata, ydata)    
    
    ax.errorbar(xdata, ydata, yerr=err)
    ax.plot(numpy.arange(101), linearFit(numpy.arange(101), 1, 0), color='black', label=r"$R^2={0:5.4f}, RMSE={1:5.4f}$".format(r2, rmse))
    ax.grid()
    ax.legend()
    ax.set(aspect='equal', xlabel="Predicted", ylabel="Actual")
    ax.axis([0, 100, 0, 100])
    ax.label_outer()
    return None
            

def correlation_values_graph(input_data, ax):
    
    corr_data = {}
    for datum in input_data:
        if round(datum[0], 1) not in corr_data:
            corr_data[round(datum[0], 1)] = []
        corr_data[round(datum[0], 1)].append(datum[1])
    xdata = []
    err = []
    ydata = []
    for datum in sorted(corr_data.keys()):
        if len(corr_data[datum]) > Globals.THRESHOLD:
            ydata.append(numpy.mean(corr_data[datum]))
            boot=bootstrap(corr_data[datum])
            err.append([ydata[-1] - boot[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                        boot[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))] - ydata[-1]])
            xdata.append(datum)
    err = numpy.transpose(err)
    xdata = numpy.array(xdata)
    ydata = numpy.array(ydata)
    rmse = RMSE(linearFit, [1, 0], xdata, ydata)
    r2 = RSquared(linearFit, [1, 0], xdata, ydata)    
    
    ax.errorbar(xdata, ydata, yerr=err)
    ax.plot(numpy.arange(-7, 7), linearFit(numpy.arange(-7, 7), 1, 0), color='black', label=r"$R^2={0:5.4f}, RMSE={1:5.4f}$".format(r2, rmse))
    ax.grid()
    ax.legend()
    ax.set(aspect='equal')
    ax.axis([-7, 7, -7, 7])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.label_outer()
    return None


def assign_from_list(outputlist, attribute):
    '''
    Lets us efficiently assign all the values from the different model outputlists to play attributes
    '''
    for game in Globals.gamelist:
        for play in game.playlist:
            setattr(play, attribute, [x.pop() for x in outputlist])
