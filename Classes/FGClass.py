# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:18:22 2018

@author: Chris Clement
"""
import math
import numpy
import scipy
import pandas
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import scipy.optimize
from sklearn.model_selection import KFold
import Functions
import Globals
import Classes.EPClass as EPClass
import traceback


class FG():
    '''
    The FG class holds all the data for FG based on field position.
    '''
    def __init__(self, ydline):
        self.YDLINE = ydline
        #self.N = self.GOOD = self.ROUGE = self.MISSED = 0  # TODO: Delete this line

        self.counts = {"GOOD" : numpy.float64(0),
                 "ROUGE" : numpy.float64(0),
                 "MISSED" : numpy.float64(0)}

        self.probabilities = {"GOOD" : numpy.array([None, None, None], dtype='Float64'),
                        "ROUGE" : numpy.array([None, None, None], dtype='Float64'),
                        "MISSED" : numpy.array([None, None, None], dtype='Float64')}

        self.EP = numpy.array([None, None, None], dtype='Float64')
        self.BOOTSTRAP = Globals.DummyArray
        self.EP_ARRAY = []

    def calculate(self):
        '''
        Calculate all the percentages and the binomial CIs.
        Can do right away b/c there's no EP aspect involved
        '''
        if sum(self.counts.values()) > 0:
            for outcome in self.counts:
                self.probabilities[outcome][1] = self.counts[outcome] / sum(self.counts.values())
                self.probabilities[outcome][2] = Functions.BinomHigh(self.counts[outcome], sum(self.counts.values()), Globals.CONFIDENCE)
                self.probabilities[outcome][0] = Functions.BinomLow(self.counts[outcome], sum(self.counts.values()), Globals.CONFIDENCE)

            self.EP[1] = sum(self.EP_ARRAY) / sum(self.counts.values())
        return None

    def wipe(self):
        '''
        Reset all the attributes because we're iterating
        '''
        counts = {"GOOD" : numpy.float64(0),
                 "ROUGE" : numpy.float64(0),
                 "MISSED" : numpy.float64(0)}

        probabilities = {"GOOD" : numpy.array([None, None, None], dtype='Float64'),
                        "ROUGE" : numpy.array([None, None, None], dtype='Float64'),
                        "MISSED" : numpy.array([None, None, None], dtype='Float64')}

        self.EP = [None, None, None]
        self.BOOTSTRAP = Globals.DummyArray
        self.EP_ARRAY = []
        return None

    def boot(self):
        '''
        Bootstrap the EP values
        '''
        if sum(self.counts.values()) > 10:
            self.BOOTSTRAP =\
                numpy.sort(numpy.array([numpy.average(numpy.random.choice(
                        self.EP_ARRAY, sum(self.counts.values()), replace=True))
                        for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))
            self.EP[0] =\
                self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)]

            self.EP[2] =\
                self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * (1-Globals.CONFIDENCE))]
        return None


FG_ARRAY = [FG(yardline) for yardline in range(110)]

FG_classification_models = []
FG_classification_models.append(sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000))
FG_classification_models.append(sklearn.neighbors.KNeighborsClassifier())
#FG_classification_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))
#FG_classification_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
#FG_classification_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))

FG_regression_models = []
#FG_regression_models.append(sklearn.linear_model.LogisticRegression(solver='saga', max_iter=10000))
FG_regression_models.append(sklearn.neighbors.KNeighborsRegressor())
FG_regression_models.append(sklearn.ensemble.RandomForestRegressor(n_estimators=Globals.forest_trees, n_jobs=-1))
FG_regression_models.append(sklearn.neural_network.MLPRegressor(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
FG_regression_models.append(sklearn.ensemble.GradientBoostingRegressor(n_estimators=Globals.forest_trees, warm_start=True))


def FG_classification():
    '''
    Building different FG models
    '''
    print("Building FG classification models", Functions.timestamp())

    for game in Globals.gamelist:
        for play in game.playlist:
            play.FG_wipe()

    FG_data = []
    FG_data_x = []
    FG_data_y = []

    for game in Globals.gamelist:
        for play in game.playlist:
            try:
                if play.FG_RSLT and play.METAR:  # Here to handle the one game with no METAR
                    # TODO: Better deal with the FGs missing wind or temp, better nan handling with pandas
                    if play.headwind is not None and play.crosswind is not None and play.METAR.temp is not None:
                        FG_data.append([play.YDLINE,
                                        game.stadium.elevation,
                                        play.METAR.temp._value,
                                        play.headwind,
                                        play.crosswind,
                                        True if play.METAR.weather else False,
                                        play.FG_RSLT])
                    else:
                        print(play.playdesc)
            except Exception as err:
                print("FG METAR ERROR", play.MULE, play.playdesc, play.METAR.string())
                print(err)
                traceback.print_exc()

    FG_data_x = pandas.DataFrame([x[:-1] for x in FG_data],
                                 columns=["ydline", "altitude", "temperature", "headwind", "crosswind", "weather"])
    FG_data_y = pandas.DataFrame([x[-1] for x in FG_data], columns=["FG_Result"])

    print(len(FG_data_y))

    for model in FG_classification_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(FG_data_y) ** 0.5)    

    outputlist = [[] for x in FG_classification_models]
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(FG_data_x)
    for train_index, test_index in kf.split(FG_data_x):
        for m, model in enumerate(FG_classification_models):
            model.fit(FG_data_x.iloc[train_index],
                      FG_data_y.iloc[train_index].values.ravel())
            outputlist[m].extend(model.predict_proba(FG_data_x.iloc[test_index]))
            print("   ", type(model).__name__, "fitted", Functions.timestamp())
    print("    KFolds fitted", Functions.timestamp())

    for model in outputlist:
        for play in model:
            play = [x for x in play]  # Converting from arrays to lists
        model.reverse()  # We need it in reverse order to be able to pop

    for game in Globals.gamelist:
        for play in game.playlist:
            if play.FG_RSLT and play.METAR:
                if play.headwind is not None and play.crosswind is not None and play.METAR.temp is not None:
                    play.FG_probs_list = [x.pop() for x in outputlist]

    for model in FG_classification_models:
        model.fit(FG_data_x, FG_data_y.values.ravel())
        print("    ", type(model).__name__, "fitted", Functions.timestamp())
    print("    Full models fitted", Functions.timestamp())

    Functions.printFeatures(FG_classification_models)
    return None


def FG_regression():
    '''
    Building different FG models
    '''
    print("Building FG regression models", Functions.timestamp())
    for game in Globals.gamelist:
        for play in game.playlist:
            play.FG_wipe()

    FG_data = []
    FG_data_x = []
    FG_data_y = []

    EP_result = 0
    for game in Globals.gamelist:
        for p, play in enumerate(game.playlist):
            try:
                if play.FG_RSLT and play.METAR:  # Here to handle the one game with no METAR
                    # TODO: Better deal with the FGs missing wind or temp, better nan handling with pandas
                    if play.headwind is not None and play.crosswind is not None and play.METAR.temp is not None:
                        if play.FG_RSLT == "GOOD":
                            EP_result = Globals.SCOREvals[0][1]
                        elif play.FG_RSLT == "ROUGE":
                            EP_result == Globals.SCOREvals[1][1]
                        elif play == game.playlist[-1]:
                            EP_result = 0
                        elif play.QUARTER == 2 and game.playlist[p+1].QUARTER == 3:
                            EP_result = 0
                        elif play.QUARTER == 4 and game.playlist[p+1].QUARTER == 5:
                            EP_result = 0
                        else:
                            EP_result = game.playlist[p+1].raw_EP[1]
                        FG_data.append([play.YDLINE,
                                        game.stadium.elevation,
                                        play.METAR.temp._value,
                                        play.headwind,
                                        play.crosswind,
                                        True if play.METAR.weather else False,
                                        EP_result])
                    else:
                        print(play.playdesc)
            except Exception as err:
                print("FG METAR ERROR", play.MULE, play.playdesc, play.METAR.string())
                print(err)
                traceback.print_exc()

    FG_data_x = pandas.DataFrame([x[:-1] for x in FG_data],
                                 columns=["ydline", "altitude", "temperature", "headwind", "crosswind", "weather"])
    FG_data_y = pandas.DataFrame([x[-1] for x in FG_data], columns=["FG_Result"])

    print(len(FG_data_y))

    for model in FG_regression_models:
        if type(model).__name__ == "KNeighborsRegressor":
            model.n_neighbors = int(len(FG_data_y) ** 0.5)    

    outputlist = [[] for x in FG_regression_models]
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(FG_data_x)
    for train_index, test_index in kf.split(FG_data_x):
        for m, model in enumerate(FG_regression_models):
            model.fit(FG_data_x.iloc[train_index],
                      FG_data_y.iloc[train_index].values.ravel())
            outputlist[m].extend(model.predict(FG_data_x.iloc[test_index]))
            print("   ", type(model).__name__, "fitted", Functions.timestamp())
    print("    KFolds fitted", Functions.timestamp())

    print(len(outputlist[0]))
    for model in outputlist:
#        for play in model:
#            play = [x for x in play]  # Converting from arrays to lists
        model.reverse()  # We need it in reverse order to be able to pop

    for game in Globals.gamelist:
        for play in game.playlist:
            if play.FG_RSLT and play.METAR:
                if play.headwind is not None and play.crosswind is not None and play.METAR.temp is not None:
                    play.FG_regression_list = [model.pop() for model in outputlist]

    for model in FG_regression_models:
        model.fit(FG_data_x, FG_data_y.values.ravel())
        print("    ", type(model).__name__, "fitted", Functions.timestamp())
    print("    Full models fitted", Functions.timestamp())

    Functions.printFeatures(FG_regression_models)
    return None


def FG_wipe():
    '''
    Tells each object in the array to reset its attributes
    TODO: Can this and those like it be put into one line?
    '''
    for YDLINE in FG_ARRAY:
        YDLINE.wipe()


def FG_calculate():
    for YDLINE in FG_ARRAY:
        YDLINE.calculate()


def FG_boot():
    print("Bootstrapping FG", Functions.timestamp())
    for YDLINE in FG_ARRAY:
        YDLINE.boot()


def FG_EP():
    '''
    Calculate EP for each FG object in the array.
    '''
    try:
        for g, game in enumerate(Globals.gamelist):
            for p, play in enumerate(game.playlist):
                if play.ODK == "FG" and play.DOWN:  # avoid PATs
                    FG_ARRAY[play.YDLINE].counts[play.FG_RSLT] += 1
                    if play.score_play:
                        FG_ARRAY[play.YDLINE].EP_ARRAY.append(Globals.score_values[play.score_play] * (1 if play.score_play_is_off else -1))
                    else:
                        for n, nextPlay in enumerate(game.playlist[p + 1:]):
                            if nextPlay.ODK == "OD":
                                FG_ARRAY[play.YDLINE].EP_ARRAY.append(EPClass.EP_ARRAY[nextPlay.DOWN][nextPlay.DISTANCE][nextPlay.YDLINE].EP[1] * (1 if nextPlay.OFFENSE == play.OFFENSE else -1))
                                break
    except Exception as err:
        print("FG EP ERROR", play.MULE, play.playdesc)
        pint(nextPlay.DOWN, nextPlay.DISTANCE, nextPlay.playdesc)
        print(err)
    return None


def FG_PLOTS():
    '''
    Puts out the graphs for field goals, probability and EP. Need to figure
    out what to do about PATs
    TODO: How about a graph showing three curves for P_GOOD, P_MISSED, P_ROUGE?
    '''

    # This is the P(FG)Good plot
    xdata = numpy.array([x.YDLINE for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD])
    ydata = numpy.array([x.P_GOOD[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD])
    error = numpy.array([[x.P_GOOD[1] - x.P_GOOD[0] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD],
            [x.P_GOOD[2] - x.P_GOOD[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD]])
    func = Functions.linearFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    plt.errorbar(xdata, ydata, yerr=error, ms=3, color='orange', fmt='v')
    plt.plot(numpy.arange(40), func(numpy.arange(40), *fit), color='orange', label=Functions.fitLabels(func).format(*fit, r2, rmse,))
    plt.xlabel("Yardline")
    plt.ylabel("$P(FG)_{Good}$")
    plt.title("$P(FG)_{Good}$ by Yardline")
    plt.grid(True)
    plt.axis([0, 40, 0, 1])
    plt.legend(loc='best')
    plt.savefig("Figures/P(FG) Good", dpi=1000)
    plt.show()

    #All three outcomes plot
    plt.errorbar(xdata, ydata, ms=3, color='blue', fmt='v', label="Good")
    ydata = numpy.array([x.P_ROUGE[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD])
    plt.errorbar(xdata, ydata, ms=3, color='red', fmt='v', label="Rouge")
    ydata = numpy.array([x.P_MISSED[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD])
    plt.errorbar(xdata, ydata, ms=3, color='yellow', fmt='v', label="Missed")
    plt.xlabel("Yardline")
    plt.ylabel("$P(FG)$")
    plt.title(("$P(FG)$ by Yardline"))
    plt.grid(True)
    plt.axis([0, 40, 0, 1])
    plt.legend(loc='best')
    plt.savefig("Figures/P(FG)", dpi=1000)
    plt.show()
    
    # EP(FG) plot
    ydata = numpy.array([x.EP[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD])
    error = numpy.array([[x.EP[1] - x.EP[0] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD],
                         [x.EP[2] - x.EP[1] for x in FG_ARRAY if sum(x.counts.values()) > Globals.THRESHOLD]])
    func = Functions.linearFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    plt.errorbar(xdata, ydata, yerr=error, ms=3, color='brown', fmt='*')
    plt.plot(numpy.arange(40), func(numpy.arange(40), *fit), color='brown', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("$EP(FG)$")
    plt.title(("$EP(FG)$ by Yardline"))
    plt.grid(True)
    plt.axis([0, 40, -1, 4])
    plt.legend(loc='best')
    plt.savefig("Figures/EP/EP(FG)", dpi=1000)
    plt.show()
    return None


def FG_classification_correlation():
    corr_graph = [[[0, 0] for x in range(110)] for model in FG_classification_models]
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.FG_probs_list:
                for m, model in enumerate(corr_graph):
                    for r, result in enumerate(["GOOD", "MISSED", "ROUGE"]):
                        model[int(round(play.FG_probs_list[m][r] * 100))][0] += 1
                        if play.FG_RSLT == result:
                            model[int(round(play.FG_probs_list[m][r] * 100))][1] += 1
                        
    for m, model in enumerate(corr_graph):
        xdata = numpy.array([x for x in numpy.arange(110) if model[x][0] > Globals.THRESHOLD])
        ydata = numpy.array([x[1]/x[0] * 100 for x in model if x[0] > Globals.THRESHOLD])
        error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model if x[0] > Globals.THRESHOLD],
                 [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model if x[0] > Globals.THRESHOLD ]]
        rmse = Functions.RMSE(Functions.linearFit, [1, 0], xdata, ydata)
        r2 = Functions.RSquared(Functions.linearFit, [1, 0], xdata, ydata)
        
        plt.plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                 label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, r2,))
        plt.errorbar(xdata, ydata, yerr=error)
        plt.legend()
        plt.title("Correlation Graph for $P(FG)$,\n" + type(FG_classificaiton_models[m]).__name__)
        plt.xlabel("Predicted $P(FG)$")
        plt.ylabel("Actual $P(FG)$")
        plt.axis([0, 100, 0, 100])
        plt.grid()
        plt.savefig("Figures/FG/FG Correlation(" + type(FG_classification_models[m]).__name__ + ")", dpi=1000)
        plt.show()    
    return None


def FG_regression_correlation():
    corr_graph = [[[0, 0] for x in range(110)] for model in FG_regression_models]
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.FG_probs_list:
                for m, model in enumerate(corr_graph):
                    for r, result in enumerate(["GOOD", "MISSED", "ROUGE"]):
                        model[int(round(play.FG_probs_list[m][r] * 100))][0] += 1
                        if play.FG_RSLT == result:
                            model[int(round(play.FG_probs_list[m][r] * 100))][1] += 1
                        
    for m, model in enumerate(corr_graph):
        xdata = numpy.array([x for x in numpy.arange(110) if model[x][0] > Globals.THRESHOLD])
        ydata = numpy.array([x[1]/x[0] * 100 for x in model if x[0] > Globals.THRESHOLD])
        error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model if x[0] > Globals.THRESHOLD],
                 [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model if x[0] > Globals.THRESHOLD ]]
        rmse = Functions.RMSE(Functions.linearFit, [1, 0], xdata, ydata)
        r2 = Functions.RSquared(Functions.linearFit, [1, 0], xdata, ydata)
        
        plt.plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                 label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, r2,))
        plt.errorbar(xdata, ydata, yerr=error)
        plt.legend()
        plt.title("Correlation Graph for $P(FG)$,\n" + type(FG_regression_models[m]).__name__)
        plt.xlabel("Predicted $P(FG)$")
        plt.ylabel("Actual $P(FG)$")
        plt.axis([0, 100, 0, 100])
        plt.grid()
        plt.savefig("Figures/FG/FG Correlation(" + type(FG_regression_models[m]).__name__ + ")", dpi=1000)
        plt.show()    
    return None

