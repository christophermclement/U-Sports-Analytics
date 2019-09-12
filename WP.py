# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:08:47 2019

@author: Chris Clement
"""

import sklearn
from sklearn.model_selection import KFold
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import sklearn.svm
import numpy
import matplotlib.pyplot as plt
import pandas
import traceback
import Globals
import Functions

'''
Here are all the models in the list, we can add whatever ones we want or take them away. They're pickled individually because otherwise it throws a memory error.
It actually throws that mamory error because of the RF model, which is somehow >4GB and pickle/my computer throws a fit. Open to suggestions?
'''

WP_classification_models = []
WP_classification_models.append(sklearn.linear_model.LogisticRegression(solver="liblinear"))
WP_classification_models.append(sklearn.neighbors.KNeighborsClassifier())
#WP_classification_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))  # TODO: Find a way to pickle this
#WP_classification_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
#WP_classification_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))


def WP_classification():
    '''
    Building the different WP models
    N.B.: I have a cheap boolean in Globals that I can flip so that my models train with lower parameters so I can do test runs, 
    that's why there are so many references to Globals.XXX, because if that boolean is True it makes the models much more refined
    (more trees, tighter tolerances, deeper nets, etc), and if it's False the models train much faster because at my core I still
    code by slapping some lines on there and hitting F5 and then squashing bugs as I go.
    TODO: Yes, this method has become way too long, I know it needs to be refactored into several sections, it's a legacy thing I haven't gotten around to.
    '''
    print("calculating WP", Functions.timestamp())  # Just lets me know how long the script has been running and where we're at with it.
    
    for game in Globals.gamelist:  # Just makes sure I'm not carrying residual garbage in these attributes
        for play in game.playlist:
            play.WP_wipe()
    
    WP_data = []
    WP_data_x = []
    WP_data_y = []

    for game in Globals.gamelist:  # Here are my features
        for play in game.playlist:
            WP_data.append([play.DOWN,
                            play.DISTANCE,
                            play.YDLINE,
                            play.TIME,
                            play.O_SCORE,
                            play.D_SCORE,
                            play.O_LEAD,
                            play.O_TO,
                            play.D_TO,
                            play.OffIsHome,
                            play.O_WIN])  # Response variable
    WP_data_x = [x[:-1] for x in WP_data]  # Separate the input and output
    WP_data_y = [x[-1] for x in WP_data]


    WP_data_x = pandas.DataFrame(WP_data_x,
                                 columns=["Down", "Distance", "Ydline", "Time",
                                          "Offense Score", "Defense Score", "O Lead",
                                          "Offense TO", "Defense TO", "OffIsHome"])  # Convert to dataframe to play nice with sklearn
    WP_data_y = pandas.DataFrame(WP_data_y, columns=["O Win"])

    for model in WP_classification_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(WP_data_y) ** 0.5)  # Typical to use sqrt(N) for kNN

    '''
    Ok, here's what's going on here: We want to do k-fold CV. Outputlist holds all the output from the models so that we can
    reassign it to the plays in the database. This way we predict each play based on the model from its fold.
    Mathematically it's really elegant but the code is ugly.
    '''
    outputlist = Functions.fit_models(WP_classification_models, WP_data_x, WP_data_y, 2)
    
    #Manipulating outputlist to get just P(win) and flip the order so we can pop from the end of the list
    outputlist = numpy.flip(numpy.take(outputlist, 1, axis=2), axis=1)
    outputlist = outputlist.tolist()
    for game in Globals.gamelist:
        for play in game.playlist:
            play.WP_list = [x.pop() for x in outputlist]

    Functions.printFeatures(WP_classification_models)  # Just prints out coefficients and such

    # This just tells each game to calculate the WPA of each play by taking the difference between each play and the next.
    # It's a game method because it's tidier that way. Inasmuch as possible the modification of play attributes is done by functions
    # in the play object, if not then as a function of the game object
    for game in Globals.gamelist:
        game.WPA_FN()
    return None 


def WP_correlation():
    '''
    Gives us the correlation graphs for our WP models
    '''
    data = []
    for game in Globals.gamelist:
        for play in game.playlist:
            data.append([play.WP_list, [play.O_WIN, play.QUARTER, play.DOWN, play.OffIsHome]])

    fig, ax = plt.subplots(figsize=(fig.nrows*5, fig.ncols*5))
    for m, model in enumerate(WP_classification_models):
        Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data], ax)
        fig.suptitle("Correlation Graph for Win Probability,\n" + type(model).__name__)
        fig.savefig("Figures/WP/WP Correlation(" + type(model).__name__+ ")", dpi=1000)
        plt.show()

    figs, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(figs.nrows*5, figs.ncols*5))    
    for m, model in enumerate(WP_classification_models):
        for qtr, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][1] == qtr + 1], ax)
            ax.set_title(Functions.ordinals(qtr + 1))
        figs.suptitle("Correlation Graph for Win Probability,\n" + type(model).__name__ + ", by quarter")
        figs.savefig("Figures/WP/WP Correlation(" + type(model).__name__ + ", by quarter", dpi=1000)
        plt.show()
        
    figs, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(figs.nrows*5, figs.ncols*5))    
    for m, model in enumerate(WP_classification_models):
        for down, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][2] == down + 1], ax)
            ax.set_title(Functions.ordinals(down + 1))
        figs.suptitle("Correlation Graph for Win Probability,\n" + type(model).__name__ + ", by down")
        figs.savefig("Figures/WP/WP Correlation(" + type(model).__name__ + ", by down", dpi=1000)
        plt.show()

    figs, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(figs.nrows*5, figs.ncols*5))    
    for m, model in enumerate(WP_classification_models):
        for home, ax in enumerate(figs.get_axes()):
            Functions.correlation_graph([[datum[0][m], datum[1][0]] for datum in data if datum[1][3] == home], ax)
            ax.set_title("Home" if home else "Away")
        figs.suptitle("Correlation Graph for Win Probability,\n" + type(model).__name__ + ", by Home/Away")
        figs.savefig("Figures/WP/WP Correlation(" + type(model).__name__ + ", by Home/Away", dpi=1000)
        plt.show()


def WP_PLOTS():
    '''
    This one shows the impact of different lead sizes on WP.
    '''
    for m, model in enumerate(WP_classification_models):
        for lead in range(-6, 7, 3):
            xdata = [x[1] for x in model.predict_proba([[
                    1,
                    10,
                    75,
                    time,
                    21,
                    (21 - lead),
                    lead,
                    2,
                    2,
                    True] for time in range(1801)])]
            plt.plot(xdata, label=lead)
        plt.title("WP by Lead and Time,\n1st & 10, -35, Home Score = 14, OffIsHome = True\n" + type(model).__name__)
        plt.xlabel("Time (s)")
        plt.ylabel("WP")
        plt.axis([0, 1800, 0, 1])
        plt.legend(loc='best')
        plt.grid()
        plt.savefig("Figures/WP/WP by Lead and Time (" + type(WP_classification_models[m]).__name__ + ")", dpi = 1000)
        plt.show()
