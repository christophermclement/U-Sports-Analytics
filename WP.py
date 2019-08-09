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

WP_models = []
WP_models.append(sklearn.linear_model.LogisticRegression(solver="liblinear"))
WP_models.append(sklearn.neighbors.KNeighborsClassifier())
#WP_models.append(sklearn.ensemble.RandomForestClassifier(n_estimators=Globals.forest_trees, n_jobs=-1))  # TODO: Find a way to pickle this
#WP_models.append(sklearn.neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=Globals.neural_network, warm_start=True))
#WP_models.append(sklearn.ensemble.GradientBoostingClassifier(n_estimators=Globals.forest_trees, warm_start=True))


def WP_Models():
    '''
    Building the different WP models
    N.B.: I have a cheap boolean in Globals that I can flip so that my models train with lower parameters so I can do test runs, 
    that's why there are so many references to Globals.XXX, because if that boolean is True it makes the models much more refined
    (more trees, tighter tolerances, deeper nets, etc), and if it's False the models train much faster because at my core I still
    code by slapping some lines on there and hitting F5 and then squashing bugs as I go.
    TODO: Yes, this method has become way too long, I know it needs to be refactored into several sections, it's a legacy thing I haven't gotten around to.
    '''
    print("calculating WP", Functions.timestamp())  # Just lets me know how long the script has been running and where we're at with it.
    global WP_models
    
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

    for model in WP_models:
        if type(model).__name__ == "KNeighborsClassifier":
            model.n_neighbors = int(len(WP_data_y) ** 0.5)  # Typical to use sqrt(N) for kNN

    '''
    Ok, here's what's going on here: We want to do k-fold CV. Outputlist holds all the output from the models so that we can
    reassign it to the plays in the database. This way we predict each play based on the model from its fold.
    Mathematically it's really elegant but the code is ugly.
    '''
    outputlist = numpy.empty((len(WP_models), 0, 2))
    kf = KFold(n_splits=Globals.KFolds)
    kf.get_n_splits(WP_data_x)
    for train_index, test_index in kf.split(WP_data_x):
        temp = []
        for m, model in enumerate(WP_models):
            model.fit(WP_data_x.iloc[train_index], WP_data_y.iloc[train_index].values.ravel())
            temp.append(model.predict_proba(WP_data_x.iloc[test_index]))
            print("\t", type(model).__name__, "fitted", Functions.timestamp())
        outputlist = numpy.concatenate((outputlist, temp), axis = 1)
    print("\tmodels fitted", Functions.timestamp())
    print([model[:25] for model in outputlist])  # prints outputlist so we can see if it's a list or what, and then compare to what it looks like after cleanup
    # Here we're refitting the models on all the data so we can use it for future prediction'
    for model in WP_models:
        model.fit(WP_data_x, WP_data_y.values.ravel())
    
    #Manipulating outputlist to get just P(win) and flip the order so we can pop from the end of the list
    outputlist = numpy.flip(numpy.take(outputlist, 1, axis=2), axis=1)
    outputlist = outputlist.tolist()
    for game in Globals.gamelist:
        for play in game.playlist:
            play.WP_list = [x.pop() for x in outputlist]

    Functions.printFeatures(WP_models)  # Just prints out coefficients and such

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
    corr_graph = [[[0, 0] for x in range(101)] for model in WP_models]

    try:
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE and play.DOWN:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.WP_list[m] * 100))][0] += 1
                        model[int(round(play.WP_list[m] * 100))][1] += play.O_WIN
    except Exception as err:
        print("corr_graph error")
        print(err)
        print(play.MULE, play.DOWN, play.playdesc)
        print(play.WP_list)
        traceback.print_exc()

    #General WP correlation
    for m, model in enumerate(corr_graph):
        xdata = numpy.arange(101)
        ydata = numpy.array([x[1]/x[0] * 100 for x in model])
        error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model],
                 [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model]]
        
        rmse = Functions.RMSE(Functions.linearFit, (1, 0), xdata, ydata)
        r2 = Functions.RSquared(Functions.linearFit, (1, 0), xdata, ydata)
        
        plt.plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                 label="RMSE={0:5.4g}, R^2={1:5.4g}".format(rmse, r2,))
        plt.errorbar(xdata, ydata, yerr=error)
        plt.legend()
        plt.title("Correlation Graph for Win Probability,\n" + type(WP_models[m]).__name__)
        plt.xlabel("Predicted WP")
        plt.ylabel("Actual WP")
        plt.axis([0, 100, 0, 100])
        plt.grid()
        plt.savefig("Figures/WP/WP Correlation(" + type(WP_models[m]).__name__ + ")", dpi=1000)
        plt.show()    

    '''
    This is a test of doing it in an OOP-based way
    '''
    axs= [[], [], [], [], []]
    fig, ([axs[1], axs[2]], [axs[3], axs[4]]) = plt.subplots(2, 2, sharex=True, sharey=True)

    corr_graph = [[[[0, 0] for wp in range(101)] for model in WP_models] for quarter in range(5)]
    for game in Globals.gamelist:
        for play in game.playlist:
            for m, model in enumerate(play.WP_list):
                corr_graph[play.QUARTER][m][int(round(play.WP_list[m] * 100))][0] += 1
                corr_graph[play.QUARTER][m][int(round(play.WP_list[m] * 100))][1] += play.O_WIN

        for m, model in enumerate(corr_graph):
            xdata = numpy.arange(101)
            ydata = numpy.array([x[1] / x[0] * 100 for x in model])
            error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model],
                     [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model]]

            axs[quarter].plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(
                             Functions.RMSE(Functions.linearFit, [1, 0], xdata, ydata),
                             Functions.RSquared(Functions.linearFit, [1, 0], xdata, ydata)))
            axs[quarter].errorbar(xdata, ydata, yerr=error, fmt='h', ms=1)

            #ax.set_title(Functions.ordinals(quarter) + " quarter")
            #ax.axis([0, 100, 0, 100])
            #ax.grid()
        #fig.xlabel("Predicted WP")
        #fig.ylabel("Actual WP")
        
        fig.suptitle("Correlation Graph for Win Probability,\n" + type(WP_models[m]).__name__ + ", by quarter")
        fig.savefig("Figures/WP/WP Correlation(" + type(WP_models[m]).__name__ + ", by quarter", dpi=3000)
    plt.show()

    '''
    # WP correlation based on quarter
    for quarter in range(1, 5):
        corr_graph = [[[0, 0] for x in range(101)] for model in WP_models]

        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE and play.DOWN > 0 and play.QUARTER == quarter:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.WP_list[m] * 100))][0] += 1
                        model[int(round(play.WP_list[m] * 100))][1] += play.O_WIN

        for m, model in enumerate(corr_graph):
            xdata = numpy.arange(101)
            ydata = numpy.array([x[1] / x[0] * 100 for x in model])
            error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model],
                     [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model]]

            plt.plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(
                             Functions.RMSE(Functions.linearFit, [1, 0], xdata, ydata),
                             Functions.RSquared(Functions.linearFit, [1, 0], xdata, ydata)))
            plt.errorbar(xdata, ydata, yerr=error, fmt='h', ms=3)
            plt.legend()
            plt.title("Correlation Graph for Win Probability,\n" + type(WP_models[m]).__name__ + ", " + Functions.ordinals(quarter) + " quarter")
            plt.xlabel("Predicted WP")
            plt.ylabel("Actual WP")
            plt.axis([0, 100, 0, 100])
            plt.grid()
            plt.savefig("Figures/WP/WP Correlation(" + type(WP_models[m]).__name__ + ", " + Functions.ordinals(quarter) + " quarter", dpi=1000)
            plt.show()
    '''

    # WP correlation based on down
    for down in range(1, 4):
        [[[0, 0] for x in range(101)] for model in WP_models]
        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE and play.DOWN > 0 and play.DOWN == down:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.WP_list[m] * 100))][0] += 1
                        model[int(round(play.WP_list[m] * 100))][1] += play.O_WIN

        for m, model in enumerate(corr_graph):
            xdata = numpy.arange(101)
            ydata = numpy.array([x[1] / x[0] * 100 for x in model])
            error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model],
                     [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model]]
            func = Functions.linearFit
            fit = [1, 0]
            r2 = Functions.RSquared(func, fit, xdata, ydata)
            rmse = Functions.RMSE(func, fit, xdata, ydata)
            plt.plot(xdata, func(xdata, *fit), color='black', label=r"$R^2={0:5.4g}, RMSE={1:5.4g}$".format(r2, rmse))
            plt.errorbar(xdata, ydata, yerr=error)
            plt.legend()
            plt.title("Correlation Graph for Win Probability,\n" + type(WP_models[m]).__name__ + ", " + Functions.ordinals(down) + " down")
            plt.xlabel("Predicted WP")
            plt.ylabel("Actual WP")
            plt.axis([0, 100, 0, 100])
            plt.grid()
            plt.savefig("Figures/WP/WP Correlation(" + type(WP_models[m]).__name__ + ", " + Functions.ordinals(down) + " down" , dpi=1000)
            plt.show()    
    
    # WP Correlation based on home/away
    for OffIsHome in range(0, 2):
        [[[0, 0] for x in range(101)] for model in WP_models]

        for game in Globals.gamelist:
            for play in game.playlist:
                if play.DISTANCE and play.DOWN > 0 and play.OffIsHome == OffIsHome:
                    for m, model in enumerate(corr_graph):
                        model[int(round(play.WP_list[m] * 100))][0] += 1
                        model[int(round(play.WP_list[m] * 100))][1] += play.O_WIN

        for m, model in enumerate(corr_graph):
            xdata = numpy.arange(101)
            ydata = numpy.array([x[1] / x[0] * 100 for x in model])
            error = [[(x[1]/x[0] - Functions.BinomLow(x[1], x[0], Globals.CONFIDENCE)) * 100 for x in model],
                     [(Functions.BinomHigh(x[1], x[0], Globals.CONFIDENCE) - x[1]/x[0]) * 100 for x in model]]

            plt.plot(xdata, Functions.linearFit(xdata, 1, 0), color='black',
                     label="RMSE={0:5.4g}, R^2={1:5.4g}".format(
                             Functions.RMSE(Functions.linearFit, [1, 0],
                                            xdata, ydata),
                             Functions.RSquared(Functions.linearFit, [1, 0],
                                                xdata, ydata)))
            plt.errorbar(xdata, ydata, yerr=error)
            plt.legend()
            plt.title("Correlation Graph for Win Probability,\n" + type(WP_models[m]).__name__ + ", OffIsHome = " + ("True" if OffIsHome else "False"))
            plt.xlabel("Predicted WP")
            plt.ylabel("Actual WP")
            plt.axis([0, 100, 0, 100])
            plt.grid()
            plt.savefig("Figures/WP/WP Correlation(" + type(WP_models[m]).__name__ + ", " + Functions.ordinals(down) + " down" , dpi=1000)
            plt.show()    


def WP_PLOTS():
    '''
    This one shows the impact of different lead sizes on WP.
    '''
    global WP_models
    for m, model in enumerate(WP_models):
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
        plt.savefig("Figures/WP/WP by Lead and Time (" + type(WP_models[m]).__name__ + ")", dpi = 1000)
        plt.show()
