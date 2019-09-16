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
import gc
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

        self.EP_regression_list = []
        self.EP_classification_list = []  # These are the actual output probabilities for the classification models
        self.EP_classification_values = []  # These are the probabilities converted to EP values. 

    def binom(self):
        '''
        Finds the binomial confidence intervals for each score probability
        TODO: This has to be converted to a dict, it's illogical to have this as a list.
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
            self.BOOTSTRAP = Functions.bootstrap(
                             [y for x in [self.Score_Counts[x] * [Globals.score_values[x[0]][1] * (1 if x[1] else -1)] for x in self.Score_Counts] for y in x])

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

EP_ARRAY = [[[EP(down, distance, yardline) for yardline in range(110)] for distance in range(110)] for down in range(4)]

EP_classification_models = []
#EP_classification_models.append(sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000))
#EP_classification_models.append(sklearn.neighbors.KNeighborsClassifier())
#EP_classification_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))
EP_classification_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
EP_classification_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))

EP_regression_models = []
#EP_regression_models.append(sklearn.linear_model.LogisticRegression(solver='saga', max_iter=10000))
EP_regression_models.append(sklearn.neighbors.KNeighborsRegressor())
EP_regression_models.append(sklearn.ensemble.RandomForestRegressor(n_estimators=Globals.forest_trees, n_jobs=-1))
EP_regression_models.append(sklearn.neural_network.MLPRegressor(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
EP_regression_models.append(sklearn.ensemble.GradientBoostingRegressor(n_estimators=Globals.forest_trees, warm_start=True))



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
            play.EP_regression_list = []
            

    for down in EP_ARRAY:
        for distance in down:
            for yardline in distance:
                yardline.EP_regression_list = []
    
    for model in EP_regression_models:
        if type(model).__name__ == "KNeighborsRegressor":
            model.n_neighbors = int(len(EP_data) ** 0.5)    

    EP_data_x = pandas.DataFrame([x[:-1] for x in EP_data], columns=["Down", "Distance", "Ydline"])
    EP_data_y = pandas.DataFrame([x[-1] for x in EP_data], columns=["EP_result"])

    outputlist = Functions.fit_models(EP_regression_models, EP_data_x, EP_data_y, 1)
    outputlist = numpy.flip(outputlist, axis=1).tolist()

    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_regression_list = [model.pop() for model in outputlist]
            EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].EP_regression_list.append(play.EP_regression_list)            

    EP_data_x = []
    outputlist = []
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_regression_list:
                    ydline.EP_regression_list = numpy.mean(numpy.array(ydline.EP_regression_list), axis = 0)
                elif ydline.DISTANCE > ydline.YDLINE or ydline.YDLINE - ydline.DISTANCE < 100:
                    ydline.EP_regression_list = numpy.full(len(EP_regression_models), numpy.nan)
                else:
                    EP_data_x.append([ydline.DOWN, ydline.DISTANCE, ydline.YDLINE])
    outputlist = numpy.flip(numpy.array([model.predict(EP_data_x).tolist() for model in EP_regression_models]), axis=1).tolist()
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_regression_list == []:
                    ydline.EP_regression_list = [model.pop() for model in outputlist]

    print("\tArray populated", Functions.timestamp())

    Functions.printFeatures(EP_regression_models)


def EP_classification():
    '''
    Building the different EP classification models

    '''
    print("Building EP classification models", Functions.timestamp())
    print("\tEP classification models:", [type(model).__name__ for model in EP_classification_models])

    EP_data = []
    EP_data_x = []
    EP_data_y = []

    for game in Globals.gamelist:
        for play in game.playlist:
            EP_data.append([play.DOWN, play.DISTANCE, play.YDLINE, play.next_score + str(play.next_score_is_off)])
            play.EP_classification_list = []
            play.EP_classification_values = []
    
    for down in EP_ARRAY:
        for distance in down:
            for yardline in distance:
                yardline.EP_classification_list = []
                yardline.EP_classification_values = []

    EP_data_x = pandas.DataFrame([x[:-1] for x in EP_data], columns=["Down", "Distance", "Ydline"])
    EP_data_y = pandas.DataFrame([x[-1] for x in EP_data], columns=["EP_Input"])

    for model in EP_classification_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(EP_data) ** 0.5)    

    outputlist = Functions.fit_models(EP_classification_models, EP_data_x, EP_data_y, 9)
    outputlist = numpy.flip(outputlist, axis=1)
    outputlist = outputlist.tolist()
    
    for game in Globals.gamelist:
        for play in game.playlist:
            play.EP_classification_list = [x.pop() for x in outputlist]
            play.EP_classification_values = [sum([prob * Globals.score_values[score[0]][1] * (1 if score[1] else -1) for prob, score in zip(model, Globals.alpha_scores)]) for model in play.EP_classification_list]
            EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].EP_classification_list.append(play.EP_classification_list)

    EP_data_x = []
    outputlist = []
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_classification_list:
                    ydline.EP_classification_list = numpy.mean(numpy.array(ydline.EP_classification_list), axis = 0).tolist()
                elif ydline.DISTANCE > ydline.YDLINE or ydline.YDLINE - ydline.DISTANCE < 100:
                    ydline.EP_classification_list = numpy.full((len(EP_classification_models), 9), numpy.nan)
                else:
                    EP_data_x.append([ydline.DOWN, ydline.DISTANCE, ydline.YDLINE])
                ydline.EP_classification_values = [sum([prob * Globals.score_values[score[0]][1] * (1 if score[1] else -1) for prob, score in zip(model, Globals.alpha_scores)]) for model in play.EP_classification_list]
    outputlist = numpy.flip(numpy.array([model.predict_proba(EP_data_x).tolist() for model in EP_classification_models]), axis=1).tolist()
    for down in EP_ARRAY:
        for distance in down:
            for ydline in distance:
                if ydline.EP_classification_list == []:
                    ydline.EP_classification_list = [model.pop() for model in outputlist]
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
    return None
        

def EP_calculate():
    '''
    Tells each EP Object to calculate it's own EP value
    '''
    for down in EP_ARRAY:
        for distance in down:
            for YDLINE in distance:
                YDLINE.calculate()
    return None


def BOOTSTRAP():
    '''
    Tells each EP object to run its own bootstrap
    '''
    print("Bootstrapping EP", Functions.timestamp())
    for down in EP_ARRAY:
        for distance in down:
            for YDLINE in distance:
                YDLINE.boot()
    return None


def raw_EP_plots():
    '''
    creates the line graph for 1st and the heatmap for 2nd and 3rd down EP
    '''
    print("Building raw EP graphs", Functions.timestamp())

    xdata = numpy.arange(1, 110)
    ydata=[]
    err=[]
    
    for YDLINE in range(1, 10):
        ydata.append(EP_ARRAY[1][YDLINE][YDLINE].EP[1])
        err.append([EP_ARRAY[1][YDLINE][YDLINE].EP[1] - EP_ARRAY[1][YDLINE][YDLINE].EP[0], 
                    EP_ARRAY[1][YDLINE][YDLINE].EP[2] - EP_ARRAY[1][YDLINE][YDLINE].EP[1]])
    for YDLINE in EP_ARRAY[1][10][10:]:
        ydata.append(YDLINE.EP[1])
        err.append([YDLINE.EP[1] - YDLINE.EP[0], YDLINE.EP[2] - YDLINE.EP[1]])
    xdata = numpy.array(xdata)
    ydata = numpy.array(ydata)
    err = numpy.transpose(err)

    fit = scipy.optimize.curve_fit(Functions.linearFit, xdata, ydata)[0]
    rmse = Functions.RMSE(Functions.linearFit, fit, xdata, ydata)
    R2 = Functions.RSquared(Functions.linearFit, fit, xdata, ydata)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, yerr=err, fmt='x', color='green', ms=3)
    plt.plot(numpy.arange(110), Functions.linearFit(numpy.arange(110), *fit),
             color='green', label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}".format(*fit, rmse, R2))
    ax.set(xlabel="Yardline", ylabel="EP")
    ax.grid(True)
    ax.legend()
    ax.axis([0, 110, -2, 7])
    fig.suptitle("EP for 1st & 10 by Yardline, Raw Data")
    fig.savefig("Figures/EP/EP(1st&10), Raw Data", dpi=1000)
    plt.close('all')
    gc.collect()
    
    # Here's the heatmap with the raw data
    for down in [2, 3]:
        heatmap_data = []
        for distance in EP_ARRAY[down]:
            temp = []
            for ydline in distance[1:]:
                if ydline.DISTANCE <= ydline.YDLINE and ydline.YDLINE - ydline.DISTANCE < 100:
                    if ydline.EP[1] is None:
                        temp.append(numpy.nan)
                    else:
                        temp.append(ydline.EP[1])
                else:
                    temp.append(numpy.nan)
            heatmap_data.append(temp)

        fig, ax = plt.subplots(1, 1, figsize = (5, 3))
        mappable = ax.imshow(heatmap_data, origin='lower', aspect=2, cmap='viridis',
                   vmin=Globals.score_values["TD"][1] * (-1),
                   vmax=Globals.score_values["TD"][1])
        fig.suptitle("EP for " + Functions.ordinals(down)
                  + " Down by Distance and Yardline,\nRaw Data")
        ax.set(xlabel="Yardline", ylabel="Distance")
        ax.grid()
        ax.axis([0, 110, 0, 25])
        fig.colorbar(mappable, ax=ax)
        fig.savefig("Figures/EP/Raw EP(" + str(down) + " down) Raw", dpi=1000)
        plt.close('all')
        gc.collect()
    return None


def EP_regression_plots():
    '''
    making the EP plots for the different regression models. Basically we're cribbing the raw code, but without consideration for confidence
    '''
    print("Building EP regression graphs", Functions.timestamp())

    for m, model in enumerate(EP_regression_models):
        print("\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        xdata = numpy.arange(1, 110)
        ydata=[]
    
        for YDLINE in range(1, 10):
            ydata.append(EP_ARRAY[1][YDLINE][YDLINE].EP[1])
        for YDLINE in EP_ARRAY[1][10][10:]:
            ydata.append(YDLINE.EP_regression_list[m])
        xdata = numpy.array(xdata)
        ydata = numpy.array(ydata)

        fit = scipy.optimize.curve_fit(Functions.linearFit, xdata, ydata)[0]
        rmse = Functions.RMSE(Functions.linearFit, fit, xdata, ydata)
        R2 = Functions.RSquared(Functions.linearFit, fit, xdata, ydata)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.errorbar(xdata, ydata, fmt='x', color='green', ms=3)
        plt.plot(numpy.arange(110), Functions.linearFit(numpy.arange(110), *fit),
                 color='green', label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}".format(*fit, rmse, R2))
        ax.set(xlabel="Yardline", ylabel="EP")
        ax.grid(True)
        ax.legend()
        ax.axis([0, 110, -2, 7])
        fig.suptitle("EP for 1st & 10 by Yardline,\n" + type(model).__name__)
        fig.savefig("Figures/EP/EP(1st&10), " + type(model).__name__, dpi=1000)
        plt.close('all')
        gc.collect()

        # Heatmaps for later downs
        for down in [2, 3]:
            heatmap_data = numpy.array([[yardline.EP_regression_list[m] for yardline in distance] for distance in EP_ARRAY[down]])
            fig, ax = plt.subplots(1, 1, figsize = (5, 3))
            mappable = ax.imshow(heatmap_data, origin='lower', aspect=2, cmap='viridis',
                       vmin=Globals.score_values["TD"][1] * (-1),
                       vmax=Globals.score_values["TD"][1])
            fig.suptitle("EP for " + Functions.ordinals(down)
                      + " Down by Distance and Yardline,\n" + type(model).__name__)
            ax.set(xlabel="Yardline", ylabel="Distance")
            ax.grid()
            ax.axis([0, 110, 0, 25])
            fig.colorbar(mappable, ax=ax)
            fig.savefig("Figures/EP/EP(" + str(down) + " down) " + type(model).__name__, dpi=1000)
            plt.close('all')
            gc.collect()
    return None


def EP_classification_plots():
    '''
    making the EP plots for the different classification models. Basically we're cribbing the raw code, but without consideration for confidence
    '''
    print("Building EP classification correlation graphs", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        xdata = numpy.arange(1, 110)
        ydata=[]
    
        for YDLINE in range(1, 10):
            ydata.append(EP_ARRAY[1][YDLINE][YDLINE].EP[1])
        for YDLINE in EP_ARRAY[1][10][10:]:
            ydata.append(YDLINE.EP_classification_values[m])
        xdata = numpy.array(xdata)
        ydata = numpy.array(ydata)

        fit = scipy.optimize.curve_fit(Functions.linearFit, xdata, ydata)[0]
        rmse = Functions.RMSE(Functions.linearFit, fit, xdata, ydata)
        R2 = Functions.RSquared(Functions.linearFit, fit, xdata, ydata)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.errorbar(xdata, ydata, fmt='x', color='green', ms=3)
        plt.plot(numpy.arange(110), Functions.linearFit(numpy.arange(110), *fit),
                 color='green', label="y={0:5.4g}x+{1:5.4g}\nRMSE={2:5.4g}, R^2={3:5.4g}".format(*fit, rmse, R2))
        ax.set(xlabel="Yardline", ylabel="EP")
        ax.grid(True)
        ax.legend()
        ax.axis([0, 110, -2, 7])
        fig.suptitle("EP for 1st & 10 by Yardline,\n" + type(model).__name__)
        fig.savefig("Figures/EP/EP(1st&10), " + type(model).__name__, dpi=1000)
        plt.close('all')
        gc.collect()

        # Heatmaps for later downs
        for down in [2, 3]:
            heatmap_data = numpy.array([[yardline.EP_classification_values[m] for yardline in distance] for distance in EP_ARRAY[down]])
            fig, ax = plt.subplots(1, 1, figsize = (5, 3))
            mappable = ax.imshow(heatmap_data, origin='lower', aspect=2, cmap='viridis',
                       vmin=Globals.score_values["TD"][1] * (-1),
                       vmax=Globals.score_values["TD"][1])
            fig.suptitle("EP for " + Functions.ordinals(down) + " Down by Distance and Yardline,\n" + type(model).__name__)
            ax.set(xlabel="Yardline", ylabel="Distance")
            ax.grid()
            ax.axis([0, 100, 0, 25])
            fig.colorbar(mappable, ax=ax)
            fig.savefig("Figures/EP/EP(" + str(down) + " down) " + type(model).__name__, dpi=1000)
            plt.close('all')
            gc.collect()

    return None

      
def EP_regression_correlation():
    '''
    This method creates all the relevant correlation graphs for the EP regression models
    '''
    print("Building EP regression correlation graphs", Functions.timestamp())
    data = []
    for game in Globals.gamelist:
        for play in game.playlist:
            for s, score in enumerate(Globals.alpha_scores):
                data.append([play.EP_regression_list, 
                             [(Globals.score_values[play.next_score][1] * (1 if play.next_score_is_off else -1)), 
                              play.QUARTER, play.DOWN, play.OffIsHome]])
    
    print("\tBuilding graphs by model", Functions.timestamp())
    
    for m, model in enumerate(EP_regression_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data], ax)
        fig.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__)
        fig.savefig("Figures/EP/EP Correlation(" + type(model).__name__+ ")", dpi=1000)
        plt.close('all')
        gc.collect()

    print("\tBuilding graphs by quarter", Functions.timestamp())
    for m, model in enumerate(EP_regression_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))    
        for qtr, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][1] == qtr + 1], ax)
            ax.set_title(Functions.ordinals(qtr + 1))
        figs.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by quarter")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + ", by quarter", dpi=1000)
        plt.close('all')
        gc.collect()
        
    print("\tBuilding graphs by down", Functions.timestamp())
    for m, model in enumerate(EP_regression_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))    
        for down, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][2] == down + 1], ax)
            ax.set_title(Functions.ordinals(down + 1))
        figs.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by down")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + ", by down", dpi=1000)
        plt.close('all')
        gc.collect()

    print("\tBuilding graphs by Home/Away", Functions.timestamp())
    for m, model in enumerate(EP_regression_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))    
        for home, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][3] == home], ax)
            ax.set_title("Home" if home else "Away")
        figs.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by Home/Away")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + "), by Home-Away", dpi=1000)
        plt.close('all')
        gc.collect()
            

def EP_classification_correlation():
    '''
    This method creates all the relevant correlation graphs for the EP classification models based on probability
    '''
    print("Building EP classification probability correlation graphs", Functions.timestamp())

    data = []
    for game in Globals.gamelist:
        for play in game.playlist:
            for s, score in enumerate(Globals.alpha_scores):
                data.append([[x[s] for x in play.EP_classification_list], 
                             [(True if (play.next_score == score[0] and play.next_score_is_off == score[1]) else False), 
                              play.QUARTER, play.DOWN, play.OffIsHome]])
    
    print("\tBuilding graphs by model", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data], ax)
        fig.suptitle("Probability Correlation Graph for Expected Points,\n" + type(model).__name__)
        fig.savefig("Figures/EP/EP Probability Correlation(" + type(model).__name__+ ")", dpi=1000)
        plt.close('all')
        gc.collect()

    print("\tBuilding graphs by quarter", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))    
        for qtr, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][1] == qtr + 1], ax)
            ax.set_title(Functions.ordinals(qtr + 1))
        figs.suptitle("Probability Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by quarter")
        figs.savefig("Figures/EP/EP Probability Correlation(" + type(model).__name__ + ", by quarter", dpi=1000)
        plt.close('all')
        gc.collect()
        
    print("\tBuilding graphs by down", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))    
        for down, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][2] == down + 1], ax)
            ax.set_title(Functions.ordinals(down + 1))
        figs.suptitle("Probability Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by down")
        figs.savefig("Figures/EP/EP Probability Correlation(" + type(model).__name__ + ", by down", dpi=1000)
        plt.close('all')
        gc.collect()

    print("\tBuilding graphs by Home/Away", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))    
        for home, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][3] == home], ax)
            ax.set_title("Home" if home else "Away")
        figs.suptitle("Probability Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by Home/Away")
        figs.savefig("Figures/EP/EP Probability Correlation(" + type(model).__name__ + ", by Home-Away", dpi=1000)
        plt.close('all')
        gc.collect()
            

def EP_classification_values_correlation(): 
    '''
    This method creates all the relevant correlation graphs for the EP classification models based on values
    '''
    print("Building EP classification values correlation graphs", Functions.timestamp())

    data=[]
    for game in Globals.gamelist:
        for play in game.playlist:
            data.append([list(play.EP_classification_values), [(Globals.score_values[play.next_score][1] * (1 if play.next_score_is_off else -1)), play.QUARTER, play.DOWN, play.OffIsHome]])

    print("\tBuilding graphs by model", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data], ax)
        fig.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__)
        fig.savefig("Figures/EP/Correlation Graph for Expected Points," + type(model).__name__, dpi=1000)
        plt.close('all')
        gc.collect()
       
    print("\tBuilding graphs by quarter", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))    
        for qtr, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][1] == qtr + 1], ax)
            ax.set_title(Functions.ordinals(qtr + 1))
        figs.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by quarter")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + ", by quarter", dpi=1000)
        plt.close('all')
        gc.collect()
       
    print("\tBuilding graphs by down", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))    
        for down, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][2] == down + 1], ax)
            ax.set_title(Functions.ordinals(down + 1))
        figs.suptitle("Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by down")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + ", by down", dpi=1000)
        plt.close('all')
        gc.collect()

    print("\tBuilding graphs by Home/Away", Functions.timestamp())
    for m, model in enumerate(EP_classification_models):
        print("\t\tBuilding graph for " + type(model).__name__, Functions.timestamp())
        figs, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        for home, ax in enumerate(figs.get_axes()):
            Functions.correlation_values_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][3] == home], ax)
            ax.set_title("Home" if home else "Away")
        figs.suptitle("Probability Correlation Graph for Expected Points,\n" + type(model).__name__ + ", by Home/Away")
        figs.savefig("Figures/EP/EP Correlation(" + type(model).__name__ + ", by Home-Away", dpi=1000)
        plt.close('all')
        gc.collect()

