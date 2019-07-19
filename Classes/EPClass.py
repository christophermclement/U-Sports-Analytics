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
import gc
import csv
import pickle

EP_models = []
# Logit is a shitty EP model
#EP_models.append(sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000))
# k-NN is an effective and cheap EP model but we don't need it
EP_models.append(sklearn.neighbors.KNeighborsClassifier())
# RF is a shitty EP model
#EP_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))
EP_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
# GBC is a brutally slow model that's maybe a hair better than MLP, if at all
EP_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))

class EP():
    '''
    An EP object holds all the information for a certain EP state of down,
    distance, and field position/
    '''

    def __init__(self, down, distance, yardline):
        self.DOWN = down
        self.DISTANCE = distance
        self.YDLINE = yardline
        self.N = 0
        self.X = 0
        self.EP = [None, None, None]
        self.SMOOTHED = None  # TODO: Do we even use this anywhere?

        # These are in the standard order
        self.Score_Counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        self.EP_probs_list = []
        self.EP_list = []

        self.temp_EP_probs = numpy.asarray([[0 for x in range(9)] for y in EP_models], dtype=float)
        self.temp_EP = [0 for x in EP_models]
        self.count = 0
        return None

    def binom(self):
        '''
        Finds the binomial confidence intervals for each score probability
        '''
        if self.N > 0:
            for x in range(9):
                self.P_Scores[x][1] = self.Score_Counts[x] / self.N
                self.P_Scores[x][0] = Functions.BinomLow(self.Score_Counts[x], self.N, Globals.CONFIDENCE)
                self.P_Scores[x][2] = Functions.BinomHigh(self.Score_Counts[x], self.N, Globals.CONFIDENCE)
        return None

    def boot(self):
        '''
        Bootstraps a confidence interval for our EP value
        '''
        if self.N > 10:
            self.BOOTSTRAP =\
                numpy.sort(numpy.array([numpy.average(numpy.random.choice((
                          [Globals.SCOREvals[3][1]] * self.Score_Counts[8] +
                          [Globals.SCOREvals[0][1]] * self.Score_Counts[5] +
                          [Globals.SCOREvals[1][1]] * self.Score_Counts[6] +
                          [Globals.SCOREvals[2][1]] * self.Score_Counts[7] +
                          [Globals.SCOREvals[4][1]] * self.Score_Counts[4] +
                          [Globals.SCOREvals[2][1] * (-1)] * self.Score_Counts[2] +
                          [Globals.SCOREvals[1][1] * (-1)] * self.Score_Counts[1] +
                          [Globals.SCOREvals[0][1] * (-1)] * self.Score_Counts[0] +
                          [Globals.SCOREvals[3][1] * (-1)] * self.Score_Counts[3]),
                          self.N, replace=True))
                          for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))

            self.EP = [self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                       multiply_SCOREvals([prob / self.N for prob in self.Score_Counts]),
                       self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]]
        return None

    def calculate(self):
        '''
        Calculates Raw EP value
        '''
        if self.N > 0:
            self.EP[1] = multiply_SCOREvals([prob/self.N for prob in self.Score_Counts])
        return None
        

EP_ARRAY = [[[EP(down, distance, yardline) for yardline in range(110)]
            for distance in range(Globals.DISTANCE_LIMIT)]
            for down in range(4)]



def EP_Models():
    '''
    Building the different EP models
    We have logit, kNN, and RF. Going forward we can consider adding an NN
    model, or a GAM or whatever else we want.
    TODO: This is nighmarishly long. Split it into chunks.
    '''
    global EP_ARRAY
    global EP_models
    print("Building EP models", Functions.timestamp())
    print("    EP models:", [type(model).__name__ for model in EP_models])

    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_wipe()

    EP_data_x, EP_data_y = EP_training_data()
    fit_EP_models()
    assign_EP_probs()

    # Assigning values by averaging all instances
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DOWN and play.DISTANCE < Globals.DISTANCE_LIMIT + 1:
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].temp_EP_probs = list(numpy.add(EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].temp_EP_probs, play.EP_probs_list))
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].temp_EP = list(numpy.add(EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].temp_EP, play.EP_list))
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].count += 1

    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.count > 0:
                    ydline.EP_probs_list = [[x / ydline.count for x in y] for y in ydline.temp_EP_probs]
                    ydline.EP_list = [x / ydline.count for x in ydline.temp_EP]
                else:
                    for model in EP_models:
                        ydline.EP_probs_list.append([x for x in model.predict_proba(
                                [[(ydline.DOWN),
                                  ydline.DISTANCE,
                                  ydline.YDLINE]])[0]])
                    for model in ydline.EP_probs_list:
                        ydline.EP_list.append(multiply_SCOREvals(model))
    print("    Array populated", Functions.timestamp())

    Functions.printFeatures(EP_models)  # Print out coefficients and such

    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_assign()
        game.EPA_FN()
    return None


def multiply_SCOREvals(probs):
    '''
    Takes a list of score probabilities and multiplies it by score values to
    give back an EP value
    '''
    return (0
            - probs[0] * Globals.SCOREvals[0][1]
            - probs[1] * Globals.SCOREvals[1][1]
            - probs[2] * Globals.SCOREvals[2][1]
            - probs[3] * Globals.SCOREvals[3][1]
            + probs[5] * Globals.SCOREvals[0][1]
            + probs[6] * Globals.SCOREvals[1][1]
            + probs[7] * Globals.SCOREvals[2][1]
            + probs[8] * Globals.SCOREvals[3][1])


def EP_COUNT():
    '''
    Determines the scorecounts for each EP object.
    '''
    print("Counting EP", Functions.timestamp())
    scorecount_dict = {"D-FG": 0, "D-ROUGE": 1, "D-SAFETY": 2, "D-TD": 3, "HALF": 4, "O-FG": 5, "O-ROUGE": 6, "O-SAFETY": 7, "O-TD": 8}
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT:
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].N += 1
                EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].Score_Counts[scorecount_dict[play.EP_INPUT]] += 1
    return None


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

    plt.errorbar(plots[0], plots[1], yerr=plots[2:], fmt='x', color='green', ms=3)
    plt.errorbar(plots[0], plots[1], yerr=plots[2:], fmt='x', color='green', ms=3)
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
        rmse = Functions.RMSE(Functions.linearFit, fit, numpy.array(plots[0]), numpy.array(plots[1]))
        R2 = Functions.RSquared(Functions.linearFit, fit, numpy.array(plots[0]), numpy.array(plots[1]))
        plt.plot(plots[0], plots[1], 'xg', ms=3)
        plt.plot(numpy.arange(110),Functions.linearFit(numpy.arange(110), *fit), color='green',
                 label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}"
                 .format(*fit, rmse, R2))
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

        heatmaps()  # Handles downs 2-4

    return None


def heatmaps():
    '''
    Fix these up with the new subplots syntax we learned
    '''
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
                       vmin=Globals.SCOREvals[3][1] * (-1),
                       vmax=Globals.SCOREvals[3][1])
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
                   vmin=Globals.SCOREvals[3][1] * (-1),
                   vmax=Globals.SCOREvals[3][1])
        plt.title("EP for " + Functions.ordinals(down)
                  + " Down by Distance and Yardline,\nRaw Data")
        plt.xlabel("Yardline")
        plt.ylabel("Distance")
        plt.grid()
        plt.colorbar()
        plt.savefig("Figures/EP/EP(" + Functions.ordinals(down) + ") Raw",
                    dpi=1000)
        plt.show()

    return None


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
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[0][1]
                    elif play.EP_INPUT == "D-ROUGE":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[1][1]
                    elif play.EP_INPUT == "D-SAFETY":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[2][1]
                    elif play.EP_INPUT == "D-TD":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[3][1]
                    elif play.EP_INPUT == "O-FG":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[0][1]
                    elif play.EP_INPUT == "O-ROUGE":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[1][1]
                    elif play.EP_INPUT == "O-SAFETY":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[2][1]
                    elif play.EP_INPUT == "O-TD":
                        model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[3][1]

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
        func = Functions.linearFit
        rmse = Functions.RMSE(func, [1, 0], xdata, ydata)
        R2 = Functions.RSquared(func, [1, 0], xdata, ydata)
        plt.plot(numpy.arange(-10, 10), numpy.arange(-10, 10), color='black',
                 label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, R2))
        plt.plot(xdata, ydata, 'g')
        plt.legend()
        plt.title("Correlation Graph for Expected Points,\n" + type(EP_models[m]).__name__)
        plt.xlabel("Predicted EP")
        plt.ylabel("Actual EP")
        plt.axis([-4, 7, -4, 7])
        plt.grid()
        plt.savefig("Figures/EP/EP Correlation(" + type(EP_models[m]).__name__ + ")", dpi=1000)
        plt.show()

    # Make the correlation graphs by quarter
    for quarter in range(1, 5):
        corr_graph = [[[0, 0] for x in range(201)] for model in EP_models]
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE < Globals.DISTANCE_LIMIT\
                        and play.DOWN > 0 and play.QUARTER == quarter:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.EP_list[m] * 10)) + 100][0] += 1
                        if play.EP_INPUT == "D-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "D-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "D-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "D-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[3][1]
                        elif play.EP_INPUT == "O-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "O-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "O-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "O-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[3][1]

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
            func = Functions.linearFit
            R2 = Functions.RSquared(func, numpy.array([1, 0]), xdata, ydata)
            rmse = Functions.RMSE(func, numpy.array([1, 0]), xdata, ydata)
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
        corr_graph = [[[0, 0] for x in range(201)] for model in EP_models]

        # TODO: Can we rework this block with scorecount_dict?
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE < Globals.DISTANCE_LIMIT and play.DOWN == down:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.EP_list[m] * 10)) + 100][0] += 1
                        if play.EP_INPUT == "D-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "D-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "D-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "D-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] -= Globals.SCOREvals[3][1]
                        elif play.EP_INPUT == "O-FG":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[0][1]
                        elif play.EP_INPUT == "O-ROUGE":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[1][1]
                        elif play.EP_INPUT == "O-SAFETY":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[2][1]
                        elif play.EP_INPUT == "O-TD":
                            model[int(round(play.EP_list[m] * 10)) + 100][1] += Globals.SCOREvals[3][1]

        for m, model in enumerate(corr_graph):
            ydata = [ep[1]/ep[0] if ep[0] > Globals.THRESHOLD else numpy.nan for ep in model]
            xdata = numpy.arange(-100, 101) / 10
            data = []
            # TODO: There has to be an easier way to do this?
            data = list(zip(xdata, ydata))
            data = [x for x in data if numpy.isfinite(x[1])]
            xdata = [x[0] for x in data]
            ydata = [x[1] for x in data]
            func = Functions.linearFit
            R2 = Functions.RSquared(func, numpy.array([1, 0]), xdata, ydata)
            rmse = Functions.RMSE(func, numpy.array([1, 0]), xdata, ydata)
            plt.plot(numpy.arange(-10, 10), numpy.arange(-10, 10), color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, R2))
            plt.plot(xdata, ydata, 'g')
            plt.legend()
            plt.title("Correlation Graph for Expected Points,\n" + type(EP_models[m]).__name__ + ", " + Functions.ordinals(down) + " down")
            plt.xlabel("Predicted EP")
            plt.ylabel("Actual EP")
            plt.axis([-4, 7, -4, 7])
            plt.grid()
            plt.savefig("Figures/EP/EP Correlation(" + type(EP_models[m]).__name__ + ", " + Functions.ordinals(down) + " down)", dpi=1000)
            plt.show()
    return None


def teamseason():
    for m, model in enumerate(EP_models):
        gc.collect()
        for season in range(2002, 2019):
            for team in Globals.CISTeams:
                print(team, season)
                tempdata = [[0, 0], [0, 0]]
                for game in Globals.gamelist:
                    if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                        for play in game.playlist:
                            if play.OFFENSE == team and play.EPA_list:
                                if play.RP == "R":
                                    tempdata[0][0] += 1
                                    tempdata[0][1] += play.EPA_list[m]
                                elif play.RP == "P":
                                    tempdata[1][0] += 1
                                    tempdata[1][1] += play.EPA_list[m]
                if tempdata[0][0] > Globals.THRESHOLD and tempdata[1][0] > Globals.THRESHOLD:
                    Functions.imscatter(tempdata[0][1] / tempdata[0][0], tempdata[1][1] / tempdata[1][0], "Logos/" + team + " logo.png", zoom=0.0075)
        plt.xlabel("Rush EPA")
        plt.ylabel("Pass EPA")
        plt.title("Rush EPA vs Pass EPA\n" + type(EP_models[m]).__name__)
        plt.grid()
        plt.savefig(("Figures/EP/Rush EPA vs Pass EPA" + type(EP_models[m]).__name__), dpi=1000)
        plt.show()
        gc.collect()

        for season in range(2002, 2019):
            for team in Globals.CISTeams:
                print(team, season)
                tempdata = [[0, 0], [0, 0]]
                for game in Globals.gamelist:
                    if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                        for play in game.playlist:
                            if play.DEFENSE == team and play.EPA_list:
                                if play.RP == "R":
                                    tempdata[0][0] += 1
                                    tempdata[0][1] += play.EPA_list[m]
                                elif play.RP == "P":
                                    tempdata[1][0] += 1
                                    tempdata[1][1] += play.EPA_list[m]
                if tempdata[0][0] > Globals.THRESHOLD and tempdata[1][0] > Globals.THRESHOLD:
                    Functions.imscatter(tempdata[0][1] / tempdata[0][0], tempdata[1][1] / tempdata[1][0], "Logos/" + team + " logo.png", zoom=0.0075)
        plt.xlabel("Defensive Rush EPA")
        plt.ylabel("Defensive Pass EPA")
        plt.title("Defensive Rush EPA vs Pass EPA\n" + type(EP_models[m]).__name__)
        plt.grid()
        plt.savefig(("Figures/EP/Defensive Rush EPA vs Pass EPA" + type(EP_models[m]).__name__), dpi=1000)
        plt.show()
        gc.collect()

        for season in range(2002, 2019):
            for conference in Globals.CISConferences:
                print(conference, season)
                tempdata = [[0, 0], [0, 0]]
                for game in Globals.gamelist:
                    if game.game_date.year == season and game.CONFERENCE == conference:
                        for play in game.playlist:
                            if play.EPA_list:
                                if play.RP == "R":
                                    tempdata[0][0] += 1
                                    tempdata[0][1] += play.EPA_list[m]
                                elif play.RP == "P":
                                    tempdata[1][0] += 1
                                    tempdata[1][1] += play.EPA_list[m]
                if tempdata[0][0] > Globals.THRESHOLD and tempdata[1][0] > Globals.THRESHOLD:
                    Functions.imscatter(tempdata[0][1] / tempdata[0][0], tempdata[1][1] / tempdata[1][0], "Logos/" + conference + " logo.png", zoom=0.0075)
        plt.xlabel("Rush EPA")
        plt.ylabel("Pass EPA")
        plt.title("Conference Rush EPA vs Pass EPA\n" + type(EP_models[m]).__name__)
        plt.grid()
        plt.savefig(("Figures/EP/Conference Rush EPA vs Pass EPA" + type(EP_models[m]).__name__), dpi=1000)
        plt.show()
        gc.collect()


def EP_training_data():
    EP_data = []
    EP_data_x = []
    EP_data_y = []

    for game in Globals.gamelist:
        for play in game.playlist:
            EP_data.append([play.DOWN, play.DISTANCE, play.YDLINE, play.EP_INPUT])
    
    # Sets the number of neighbours equal to the square root of the number of samples in the dataset for kNN models
    for model in EP_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(EP_data) ** 0.5)    

    EP_data_x = pandas.DataFrame([x[:-1] for x in EP_data], columns=["Down", "Distance", "Ydline"])
    EP_data_y = pandas.DataFrame([x[-1] for x in EP_data], columns=["EP_Input"])
    return EP_data_x, EP_data_y


def fit_EP_models():
    outputlist = [[] for x in EP_models]
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(EP_data_x)
    for train_index, test_index in kf.split(EP_data_x):
        for m, model in enumerate(EP_models):
            model.fit(EP_data_x.iloc[train_index],
                      EP_data_y.iloc[train_index].values.ravel())
            outputlist[m].extend(EP_models[m].predict_proba(EP_data_x.iloc[test_index]))
            print("    ", type(model).__name__, "fitted", Functions.timestamp())
    print("    KFolds fitted", Functions.timestamp())

    # Refit over the entire dataset
    for model in EP_models:
        model.fit(EP_data_x, EP_data_y.values.ravel())
        print("    ", type(model).__name__, "fitted", Functions.timestamp())
    print("    Full models fitted", Functions.timestamp())
    for model in outputlist:
        for play in model:
            play = [x for x in play]  # Converting from arrays to lists
        model.reverse()  # We need it in reverse order to be able to pop
    return None


def assign_EP_probs():
    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_probs_list = [x.pop() for x in outputlist]

    for game in Globals.gamelist:
        for play in game.playlist:
            for model in play.EP_probs_list:
                play.EP_list.append(0
                                    - model[0] * Globals.SCOREvals[0][1]
                                    - model[1] * Globals.SCOREvals[1][1]
                                    - model[2] * Globals.SCOREvals[2][1]
                                    - model[3] * Globals.SCOREvals[3][1]
                                    + model[5] * Globals.SCOREvals[0][1]
                                    + model[6] * Globals.SCOREvals[1][1]
                                    + model[7] * Globals.SCOREvals[2][1]
                                    + model[8] * Globals.SCOREvals[3][1])
    return None
