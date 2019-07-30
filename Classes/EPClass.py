# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:14:03 2018

@author: Chris Clement
"""
import Functions
import Globals
import numpy
import matplotlib.pyplot as plt
import pandas
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import scipy.optimize
from sklearn.model_selection import KFold

import csv


class EP():
    '''
    An EP object holds all the information for a certain EP state of down,
    distance, and field position/
    '''

    def __init__(self, down, distance, yardline):
        self.DOWN = down
        self.DISTANCE = distance
        self.YDLINE = yardline
        self.EP = numpy.array([None, None, None], dtype = 'float')
        # self.SMOOTHED = None  # TODO: Do we even use this anywhere?

        # These are in the standard order
        self.Score_Counts = {("FG", True): 0,
                             ("FG", False): 0,
                             ("ROUGE", True): 0,
                             ("ROUGE", False): 0,
                             ("SAFETY", True): 0,
                             ("SAFETY", False): 0,
                             ("TD", True): 0,
                             ("TD", False): 0,
                             ("HALF", False): 0}
        self.P_Scores = [[None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None],
                         [None, None, None]]

        self.BOOTSTRAP = Globals.DummyArray
        # TODO: These should probably be deleted once we show that they're no longer relevant
        #self.EP_probs_list = []
        #self.EP_list = []

        self.EP_regression_list = []
        self.EP_classification_list = []  # These are the actual output probabilities for the classification models
        self.EP_classification_values = []  # These are the probabilities converted to EP values. 
        # TODO: Technically we don'the classification values since it can be derived from the list need this but it's a long one-liner.

    def binom(self):
        '''
        Finds the binomial confidence intervals for each score probability
        TODO: no more self.N
        '''
        if sum(self.Score_Counts.values()):
            for x in range(9):
                self.P_Scores[x][1] = self.Score_Counts[x] / sum(self.Score_Counts.values())
                self.P_Scores[x][0] = Functions.BinomLow(self.Score_Counts[x], sum(self.Score_Counts.values()), Globals.CONFIDENCE)
                self.P_Scores[x][2] = Functions.BinomHigh(self.Score_Counts[x], sum(self.Score_Counts.values()), Globals.CONFIDENCE)
        return None

    def boot(self):
        '''
        Bootstraps a confidence interval for our EP value
        TODO: Fix this with the new system
        '''
        if sum(self.Score_Counts.values()):
            self.BOOTSTRAP =\
                numpy.sort(
                    numpy.array([
                        numpy.average(
                         numpy.random.choice(
                             [y for x in [self.Score_Counts[x] * [Globals.score_values[x[0]][1] * (1 if x[1] else -1)] for x in self.Score_Counts] for y in x],
                             sum(self.Score_Counts.values()), replace=True)) for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='float'))

            self.EP[0] = self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)]
            self.EP[2] = self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]
        return None

    def calculate(self):
        '''
        Calculates Raw EP value
        '''
        if sum(self.Score_Counts.values()):
            self.EP[1] = sum([(self.Score_Counts[score] * Globals.score_values[score[0]][1] * (1 if score[1] else -1)) for score in self.Score_Counts]) / sum(self.Score_Counts.values())
        return None

EP_ARRAY = [[[EP(down, distance, yardline) for yardline in range(110)] for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]

EP_classification_models = []
#EP_classification_models.append(sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000))
#EP_classification_models.append(sklearn.neighbors.KNeighborsClassifier())
EP_classification_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))
#EP_classification_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
#EP_classification_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))

EP_regression_models = []
#EP_regression_models.append(sklearn.linear_model.LogisticRegression(solver='saga', max_iter=10000))
EP_regression_models.append(sklearn.neighbors.KNeighborsRegressor())
EP_regression_models.append(sklearn.ensemble.RandomForestRegressor(n_estimators=Globals.forest_trees, n_jobs=-1))
#EP_regression_models.append(sklearn.neural_network.MLPRegressor(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
#EP_regression_models.append(sklearn.ensemble.GradientBoostingRegressor(n_estimators=Globals.forest_trees, warm_start=True))



def EP_regression():
    '''
    Building the different EP regression models
    '''
    print("Building EP regression models", Functions.timestamp())
    print("\tEP regression models:", [type(model).__name__ for model in EP_regression_models])

    EP_data = []
    EP_data_x = []
    EP_data_y = []

    for game in Globals.gamelist:
        for play in game.playlist:
            EP_data.append([play.DOWN, play.DISTANCE, play.YDLINE, Globals.score_values[play.next_score][1] * (1 if play.next_score_is_off else -1)])
    
    for model in EP_regression_models:
        if type(model).__name__ == "KNeighborsRegressor":
            model.n_neighbors = int(len(EP_data) ** 0.5)    

    EP_data_x = pandas.DataFrame([x[:-1] for x in EP_data], columns=["Down", "Distance", "Ydline"])
    EP_data_y = pandas.DataFrame([x[-1] for x in EP_data], columns=["EP_result"])

    outputlist = [[] for x in EP_regression_models]
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(EP_data_x)
    for train_index, test_index in kf.split(EP_data_x):
        for m, model in enumerate(EP_regression_models):
            model.fit(EP_data_x.iloc[train_index],
                      EP_data_y.iloc[train_index].values.ravel())
            outputlist[m].extend(model.predict(EP_data_x.iloc[test_index]))
            print("\t", type(model).__name__, "fitted", Functions.timestamp())
    print("\tKFolds fitted", Functions.timestamp())

    for model in outputlist:
#        for play in model:
#            play = [x for x in play]  # Converting from arrays to lists
        model.reverse()  # We need it in reverse order to be able to pop

    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_regression_list = [model.pop() for model in outputlist]

    # Refit over the entire dataset
    for model in EP_regression_models:
        model.fit(EP_data_x, EP_data_y.values.ravel())
        print("\t", type(model).__name__, "fitted", Functions.timestamp())
    print("\tFull models fitted", Functions.timestamp())

    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DOWN and play.DISTANCE < Globals.DISTANCE_LIMIT:
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].EP_regression_list.append(play.EP_regression_list)

    EP_data_x = []
    outputlist = []
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_regression_list:
                    ydline.EP_regression_list = numpy.mean(numpy.array(ydline.EP_regression_list), axis = 0)
                else:
                    EP_data_x.append([ydline.DOWN, ydline.DISTANCE, ydline.YDLINE])
    outputlist = [model.predict(EP_data_x) for model in EP_regression_models]
    outputlist = [list(model).reverse for model in outputlist]
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_regression_list is None:
                    ydline.EP_regression_list = [model.pop() for model in outputlist]
    print("\tArray populated", Functions.timestamp())

    Functions.printFeatures(EP_regression_models)


# TODO: Now assign all the vals to the EP_ARRAY for both classification and regression
def EP_classification():
    '''
    Building the different EP classification models
    We have logit, kNN, and RF. Going forward we can consider adding an NN
    model, or a GAM or whatever else we want.
    '''
    print("Building EP classification models", Functions.timestamp())
    print("\tEP classification models:", [type(model).__name__ for model in EP_classification_models])

    EP_data = []
    EP_data_x = []
    EP_data_y = []

    for game in Globals.gamelist:
        for play in game.playlist:
            EP_data.append([play.DOWN, play.DISTANCE, play.YDLINE, play.next_score + str(play.next_score_is_off)])
    
    for model in EP_classification_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(EP_data) ** 0.5)    

    EP_data_x = pandas.DataFrame([x[:-1] for x in EP_data], columns=["Down", "Distance", "Ydline"])
    EP_data_y = pandas.DataFrame([x[-1] for x in EP_data], columns=["EP_Input"])

    outputlist = [[] for x in EP_classification_models]
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(EP_data_x)
    for train_index, test_index in kf.split(EP_data_x):
        for m, model in enumerate(EP_classification_models):
            model.fit(EP_data_x.iloc[train_index],
                      EP_data_y.iloc[train_index].values.ravel())
            outputlist[m].extend(EP_classification_models[m].predict_proba(EP_data_x.iloc[test_index]))
            print("\t", type(model).__name__, "fitted", Functions.timestamp())
    print("    KFolds fitted", Functions.timestamp())

    for model in outputlist:
        for play in model:
            play = [x for x in play]  # Converting from arrays to lists
        model.reverse()  # We need it in reverse order to be able to pop

    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_classification_list = [x.pop() for x in outputlist]

    # Refit over the entire dataset
    for model in EP_classification_models:
        model.fit(EP_data_x, EP_data_y.values.ravel())
        print("\t", type(model).__name__, "fitted", Functions.timestamp())
    print("    Full models fitted", Functions.timestamp())

    # Assigning values by averaging all instances
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DOWN and play.DISTANCE < Globals.DISTANCE_LIMIT:
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].EP_classification_list.append(play.EP_classification_list)

    EP_data_x = []
    outputlist = []
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_classification_list:
                    ydline.EP_classification_list = numpy.mean(numpy.array(ydline.EP_classification_list), axis = 0)
                else:
                    EP_data_x.append([ydline.DOWN, ydline.DISTANCE, ydline.YDLINE])
    outputlist = [model.predict_proba(EP_data_x) for model in EP_classification_models]
    outputlist = [list(model).reverse for model in outputlist]
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_classification_list is None:
                    ydline.EP_classification_list = [model.pop for model in outputlist]
            ydline.EP_classification_values = [sum([prob * Globals.score_values[score[0]] * (1 if score[1] else -1) for prob, score in zip(model, Globals.alpha_scores)]) for model in ydline.EP_classification_list]
    print("\tArray populated", Functions.timestamp())

    Functions.printFeatures(EP_classification_models)
    return None


def EP_COUNT():
    '''
    Determines the scorecounts for each EP object.
    '''
    try:
        print("Counting EP", Functions.timestamp())
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE < Globals.DISTANCE_LIMIT:
                    EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].Score_Counts[(play.next_score, play.next_score_is_off)] += 1
    except Exception as err:
        print("EP COUNT ERROR", play.MULE, play.next_score, play.next_score_is_off, play.playdesc)
        print(err)
        

def EP_calculate():
    '''
    Tells each EP Object to calculate it's own EP value
    '''
    for down in EP_ARRAY:
        for distance in down:
            for YDLINE in distance:
                YDLINE.calculate()


def BOOTSTRAP():
    '''
    Tells each EP object to run its own bootstrap
    '''
    print("Bootstrapping EP", Functions.timestamp())
    for down in EP_ARRAY:
        for distance in down:
            for YDLINE in distance:
                YDLINE.boot()


def EP_PLOTS():
    '''
    Here we make the plots for EP, with the 1st & 10 raw, 1st & 10 for each
    model, heatmaps for 2nd & 3rd, and so on. We don't do the correlation
    graphs here.

    While we should convert this to use the oneliners, it's going to be a
    hassle to deal with the &Goal aspect of it, so in the end I don't think
    we'd end up saving anything.
    '''

    # Plotting 1st & 10 EP w/ errorbars
    plots = [[], [], [], []]
    for YDLINE in range(10):
        if EP_ARRAY[1][YDLINE][YDLINE].N > 100:
            plots[0].append(EP_ARRAY[1][YDLINE][YDLINE].YDLINE)
            plots[1].append(EP_ARRAY[1][YDLINE][YDLINE].EP[1])
            plots[2].append(EP_ARRAY[1][YDLINE][YDLINE].EP[1] -
                            EP_ARRAY[1][YDLINE][YDLINE].EP[0])
            plots[3].append(EP_ARRAY[1][YDLINE][YDLINE].EP[2] -
                            EP_ARRAY[1][YDLINE][YDLINE].EP[1])

    for YDLINE in EP_ARRAY[1][10]:
        if YDLINE.N > 100:
            plots[0].append(YDLINE.YDLINE)
            plots[1].append(YDLINE.EP[1])
            plots[2].append(YDLINE.EP[1] - YDLINE.EP[0])
            plots[3].append(YDLINE.EP[2] - YDLINE.EP[1])

    # These are throwaway variables because otherwise the syntax becomes unruly
    fit = scipy.optimize.curve_fit(Functions.linearFit, plots[0], plots[1])[0]
    rmse = Functions.RMSE(Functions.linearFit, fit, numpy.array(plots[0]),
                          numpy.array(plots[1]))
    R2 = Functions.RSquared(Functions.linearFit, fit, numpy.array(plots[0]),
                            numpy.array(plots[1]))

    plt.errorbar(plots[0], plots[1], yerr=plots[2:], fmt='x', color='green',
                 ms=3)
    plt.plot(numpy.arange(110), Functions.linearFit(numpy.arange(110), *fit),
             color='green',
             label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}"
             .format(*fit, rmse, R2))
    plt.xlabel("Yardline")
    plt.ylabel("EP")
    plt.title("EP for 1st & 10 by Yardline")
    plt.grid(True)
    plt.legend()
    plt.axis([0, 110, -3, 8])
    plt.savefig("Figures/EP/EP(1st&10)", dpi=1000)
    plt.show()

    # 1st & 10 graphs for the different models
    for m, model in enumerate(EP_models):
        plots = [[], [], [], []]
        for YDLINE in range(10):
            if EP_ARRAY[1][YDLINE][YDLINE].N > Globals.THRESHOLD:
                plots[0].append(EP_ARRAY[1][YDLINE][YDLINE].YDLINE)
                plots[1].append(EP_ARRAY[1][YDLINE][YDLINE].EP_list[m])

        for YDLINE in EP_ARRAY[1][10]:
            if YDLINE.N > Globals.THRESHOLD:
                plots[0].append(YDLINE.YDLINE)
                plots[1].append(YDLINE.EP_list[m])

        fit = scipy.optimize.curve_fit(Functions.linearFit, plots[0],
                                       plots[1])[0]
        error = Functions.RMSE(Functions.linearFit, fit,
                               numpy.array(plots[0]), numpy.array(plots[1]))
        R2 = Functions.RSquared(Functions.linearFit, fit,
                                numpy.array(plots[0]), numpy.array(plots[1]))
        plt.plot(plots[0], plots[1], 'xg', ms=3)
        plt.plot(numpy.arange(110),Functions.linearFit(numpy.arange(110), *fit), color='green',
                 label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}"
                 .format(*fit, error, R2))
        plt.xlabel("Yardline")
        plt.ylabel("EP")
        plt.title("EP for 1st & 10 by Yardline\n"
                  + type(EP_models[m]).__name__)
        plt.grid(True)
        plt.legend()
        plt.axis([0, 110, -3, 8])
        plt.savefig("Figures/EP/EP(1st&10 " + type(EP_models[m]).__name__
                    + ")", dpi=1000)
        plt.show()

    # Plotting the heatmaps by down
    for down in range(2, 4):
        for m, model in enumerate(EP_models):
            heatmap_data = []
            for distance in EP_ARRAY[down][1:]:
                temp = []
                for ydline in distance[1:]:
                    if ydline.DISTANCE <= ydline.YDLINE\
                            and ydline.YDLINE - ydline.DISTANCE < 100:
                        EPscore = ydline.EP_list[m]
                    else:
                        EPscore = (numpy.nan)
                    temp.append(EPscore)
                heatmap_data.append(temp)

            plt.imshow(heatmap_data, origin='lower', aspect=2, cmap='rainbow',
                       vmin=Globals.score_values["TD"][1] * (-1),
                       vmax=Globals.score_values["TD"][1])
            plt.title("EP for " + Functions.ordinals(down)
                      + " Down by Distance and Yardline,\n"
                      + type(model).__name__)
            plt.xlabel("Yardline")
            plt.ylabel("Distance")
            plt.grid()
            plt.colorbar()
            plt.savefig("Figures/EP/EP(" + Functions.ordinals(down) + ") "
                        + type(EP_models[m]).__name__, dpi=1000)
            plt.show()

        # Here's the heatmap with the raw data
        heatmap_data = []
        for distance in EP_ARRAY[down][1:]:
            temp = []
            for ydline in distance[1:]:
                if ydline.DISTANCE <= ydline.YDLINE\
                        and ydline.YDLINE - ydline.DISTANCE < 100:
                    if ydline.EP[1] is None:
                        temp.append(numpy.nan)
                    else:
                        temp.append(ydline.EP[1])
                else:
                    temp.append(numpy.nan)
            heatmap_data.append(temp)

        plt.imshow(heatmap_data, origin='lower', aspect=2, cmap='rainbow',
                   vmin=Globals.score_values["TD"][1] * (-1),
                   vmax=Globals.score_values["TD"][1])
        plt.title("EP for " + Functions.ordinals(down)
                  + " Down by Distance and Yardline,\nRaw Data")
        plt.xlabel("Yardline")
        plt.ylabel("Distance")
        plt.grid()
        plt.colorbar()
        plt.savefig("Figures/EP/EP(" + Functions.ordinals(down) + ") Raw",
                    dpi=1000)
        plt.show()


def EP_correlation():
    '''
    Here we're trying to build an EP correlation graph to compare predicted to
    actual EP. But for WP it's easy because it spans from 0-11, which maps
    nicely to an array. Here I think it might be easier to go from -10 to 10,
    but shift it to the right and put the array from 0-10.
    '''
    global EP_ARRAY
    global EP_models

    corr_graph = [[[0, 0] for x in range(201)] for model in EP_models]

    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT and play.DOWN > 0:
                for m, model in enumerate(corr_graph):
                    model[int(round(play.EP_list[m] * 10)) + 100][0] += 1
                    if play.EP_INPUT == "D-FG":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -=\
                            Globals.SCOREvals[0][1]
                    elif play.EP_INPUT == "D-ROUGE":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -=\
                            Globals.SCOREvals[1][1]
                    elif play.EP_INPUT == "D-SAFETY":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -=\
                            Globals.SCOREvals[2][1]
                    elif play.EP_INPUT == "D-TD":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -=\
                            Globals.SCOREvals[3][1]
                    elif play.EP_INPUT == "O-FG":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] +=\
                            Globals.SCOREvals[0][1]
                    elif play.EP_INPUT == "O-ROUGE":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] +=\
                            Globals.SCOREvals[1][1]
                    elif play.EP_INPUT == "O-SAFETY":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] +=\
                            Globals.SCOREvals[2][1]
                    elif play.EP_INPUT == "O-TD":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] +=\
                            Globals.SCOREvals[3][1]

    # Make the general correlation graphs
    for m, model in enumerate(corr_graph):
        ydata = []
        for ep in model:
            if ep[0] > Globals.THRESHOLD:
                ydata.append(ep[1]/ep[0])
            else:
                ydata.append(numpy.nan)
        xdata = numpy.arange(-100, 101) / 10
        data = list(zip(xdata, ydata))
        data = [x for x in data if numpy.isfinite(x[1])]
        xdata = numpy.array([x[0] for x in data])
        ydata = numpy.array([x[1] for x in data])
        error = Functions.RMSE(Functions.linearFit, [1, 0], xdata, ydata)
        R2 = Functions.RSquared(Functions.linearFit, [1, 0], xdata, ydata)
        plt.plot(numpy.arange(-10, 10), numpy.arange(-10, 10), color='black',
                 label="RMSE={0:5.4g}, R^2={1:5.4g}".format(error, R2))
        plt.plot(xdata, ydata, 'g')
        plt.legend()
        plt.title("Correlation Graph for Expected Points,\n"
                  + type(EP_models[m]).__name__)
        plt.xlabel("Predicted EP")
        plt.ylabel("Actual EP")
        plt.axis([-4, 7, -4, 7])
        plt.grid()
        plt.savefig("Figures/EP/EP Correlation(" + type(EP_models[m]).__name__
                    + ")", dpi=1000)
        plt.show()

    # Make the correlation graphs by quarter
    for quarter in range(1, 5):
        '''
        #TODO: Replace with a list comprehension
        corr_graph = []
        for z in range(len(EP_models)):  # For the three models
            corr_graph.append([[0, 0] for x in range(201)])
        '''
        corr_graph = [[[0, 0] for x in range(201)] for model in EP_models]

        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE < Globals.DISTANCE_LIMIT\
                        and play.DOWN > 0 and play.QUARTER == quarter:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.EP_list[m] * 10)) + 100][0] += 1
                        if play.EP_INPUT == "D-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "D-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "D-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "D-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[3][1]
                        elif play.EP_INPUT == "O-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "O-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "O-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "O-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[3][1]

        for m, model in enumerate(corr_graph):
            ydata = []
            for ep in model:
                if ep[0] > Globals.THRESHOLD:
                    ydata.append(ep[1]/ep[0])
                else:
                    ydata.append(numpy.nan)
            xdata = numpy.arange(-100, 101) / 10
            data = []
            data = list(zip(xdata, ydata))
            data = [x for x in data if numpy.isfinite(x[1])]
            xdata = [x[0] for x in data]
            ydata = [x[1] for x in data]
            R2 = Functions.RSquared(Functions.linearFit, numpy.array([1, 0]),
                                    xdata, ydata)
            rmse = Functions.RMSE(Functions.linearFit, numpy.array([1, 0]),
                                  xdata, ydata)
            plt.plot(numpy.arange(-10, 10), numpy.arange(-10, 10), color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, R2))
            plt.plot(xdata, ydata, 'g')
            plt.legend()
            plt.title("Correlation Graph for Expected Points,\n"
                      + type(EP_models[m]).__name__ + ", "
                      + Functions.ordinals(quarter) + " quarter")
            plt.xlabel("Predicted EP")
            plt.ylabel("Actual EP")
            plt.axis([-4, 7, -4, 7])
            plt.grid()
            plt.savefig("Figures/EP/EP Correlation("
                        + type(EP_models[m]).__name__ + ", "
                        + Functions.ordinals(quarter) + " quarter)", dpi=1000)
            plt.show()

    # Make the correlation graphs by down
    for down in range(1, 4):
        '''
        corr_graph = []
        for z in range(len(EP_models)):  # For the three models
            corr_graph.append([[0, 0] for x in range(201)])
        '''

        corr_graph = [[[0, 0] for x in range(201)] for model in EP_models]

        # TODO: Can we rework this block with scorecount_dict?
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE < Globals.DISTANCE_LIMIT and play.DOWN == down:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.EP_list[m] * 10)) + 100][0] += 1
                        if play.EP_INPUT == "D-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "D-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "D-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "D-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                -= Globals.SCOREvals[3][1]
                        elif play.EP_INPUT == "O-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "O-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "O-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "O-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1]\
                                += Globals.SCOREvals[3][1]

        for m, model in enumerate(corr_graph):
            ydata = [ep[1]/ep[0] if ep[0] > Globals.THRESHOLD else numpy.nan for ep in model]
            xdata = numpy.arange(-100, 101) / 10
            data = []
            # TODO: There has to be an easier way to do this?
            data = list(zip(xdata, ydata))
            data = [x for x in data if numpy.isfinite(x[1])]
            xdata = [x[0] for x in data]
            ydata = [x[1] for x in data]
            R2 = Functions.RSquared(Functions.linearFit, numpy.array([1, 0]),
                                    xdata, ydata)
            rmse = Functions.RMSE(Functions.linearFit, numpy.array([1, 0]),
                                  xdata, ydata)
            plt.plot(numpy.arange(-10, 10), numpy.arange(-10, 10),
                     color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, R2))
            plt.plot(xdata, ydata, 'g')
            plt.legend()
            plt.title("Correlation Graph for Expected Points,\n"
                      + type(EP_models[m]).__name__ + ", "
                      + Functions.ordinals(down) + " down")
            plt.xlabel("Predicted EP")
            plt.ylabel("Actual EP")
            plt.axis([-4, 7, -4, 7])
            plt.grid()
            plt.savefig("Figures/EP/EP Correlation("
                        + type(EP_models[m]).__name__ + ", " +
                        Functions.ordinals(down) + " down)", dpi=1000)
            plt.show()
