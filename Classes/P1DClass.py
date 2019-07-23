<<<<<<< HEAD
<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:12:18 2018

@author: Chris Clement
"""
import Functions
import Globals
import matplotlib.pyplot as plt
import scipy.optimize
import numpy
import gc

P1D_ARRAY = []
P1D_GOAL_ARRAY = []


# This holds all the P1D data for a given D&D
class P1D():
    '''P1D objects hold information for a P1D state of down & distance
    '''

    def __init__(self, down, distance):
        self.DOWN = down
        self.DISTANCE = distance
        self.N = self.X = 0
        self.P = [None, None, None]  # Probability with confidence interval
        self.SMOOTHED = None

    def binom(self):
        if self.X > 0:
            self.P[1] = self.X / self.N
            self.P[0] = Functions.BinomLow(self.X, self.N, Globals.CONFIDENCE)
            self.P[2] = Functions.BinomHigh(self.X, self.N, Globals.CONFIDENCE)


def Array_Declaration():
    global P1D_ARRAY
    global P1D_GOAL_ARRAY
    P1D_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    P1D_GOAL_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    return None

def P1D_calculate():
    '''
    Calculate P(1D) for  all values of D&D
    '''
    print("calculating P(1D)", Functions.timestamp())
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT:
                if play.ODK == "OD":
                    if play.DISTANCE == play.YDLINE:
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT
                    else:
                        P1D_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT

    for down in P1D_ARRAY:
        for distance in down:
            distance.binom()
    for down in P1D_GOAL_ARRAY:
        for distance in down:
            distance.binom()

'''
   comment this function
   We're making all the P1D graphs, and doing it in a more organized way with
   one-liner filters. The way we had before with all the iterative loops was a
   nightmare. Eventually we need to include the plot figures themselves. We
   also need a function
'''


def P1D_PLOTS():
    '''
    Make the graphs for P(1D) for all downs
    '''
    # TODO: This is a horrifying mess, please clean up.
    P1D_goal_plot = [[], [], [], []]
    P1D_formats = [['', '', '0th', ''],
                   ['b', 'o', '1st', Functions.linearFit,
                    ((-1, 0), (0, 1.5)),
                    Functions.fitLabels(Functions.linearFit)],
                   ['r', 's', '2nd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)],
                   ['y', '^', '3rd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)]]


    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        error = numpy.array([[(x.P[2] - x.P[1]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD],
                             [(x.P[1] - x.P[0]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD]])
        fit = scipy.optimize.curve_fit(P1D_formats[down][3], xdata, ydata, bounds=P1D_formats[down][4])[0]
        r2 = Functions.RSquared(P1D_formats[down][3], fit, xdata, ydata)
        rmse = Functions.RMSE(P1D_formats[down][3], fit, xdata, ydata)
        plt.errorbar(xdata, ydata, yerr=error, ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1])
        plt.plot(numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1),
                 P1D_formats[down][3](numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1), *fit),
                 color=P1D_formats[down][0], label=P1D_formats[down][5].format(*fit, r2, rmse))
        plt.xlabel("Distance")
        plt.ylabel("P(1D)")
        plt.title(("P(1D) of " + Functions.ordinals[down] + " Down by Distance"))
        plt.grid(True)
        plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
        plt.legend(loc='best')
        plt.savefig(("Figures/P(1D)/P(1D) " + P1D_formats[down][2] + " Down"), dpi=1000)
        plt.show()

    #All downs?
    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1],
                     label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) all Downs"), dpi=1000)
    plt.show()

    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0],
                     fmt=P1D_formats[down][1], label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of & Goal for all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) &Goal"), dpi=1000)
    plt.show()
    return None

def teamseason():
    xdata = []
    ydata = []
    
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.OFFENSE == team and play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("$P(1D)_{1^{st}&10}$")
    plt.ylabel("$P(1D)_{2^{nd}&10}$")
    plt.title("$P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for defensive P(1D)
    '''
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.DEFENSE == team:
                            if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                                tempdata[play.DOWN][0] += 1
                                tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("Defensive $P(1D)_{1^{st}&10}$")
    plt.ylabel("Defensive $P(1D)_{2^{nd}&10}$")
    plt.title("Defensive $P(1D)_{1^{st}&10} vs. P(1D)_{2^{nd}&10}$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/Defensive P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for P(1D) by conference
    '''
    for season in range(2002, 2019):
        for conference in Globals.CISConferences:
            print(season, conference)
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and game.CONFERENCE == conference:
                    for play in game.playlist:
                        if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                try:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + conference + " logo.png", zoom=0.0075)
                except Exception:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/U SPORTS logo.png", zoom=0.0075)
    plt.xlabel("Conference $P(1D)_{1^{st}&10}$")
    plt.ylabel("Conference $P(1D)_{2^{nd} & 10}$")
    plt.title("Conference $P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.45, 0.7, 0.15, 0.5])
    plt.savefig(("Figures/P(1D)/Conference P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:12:18 2018

@author: Chris Clement
"""
import Functions
import Globals
import matplotlib.pyplot as plt
import scipy.optimize
import numpy
import gc

P1D_ARRAY = []
P1D_GOAL_ARRAY = []


# This holds all the P1D data for a given D&D
class P1D():
    '''P1D objects hold information for a P1D state of down & distance
    '''

    def __init__(self, down, distance):
        self.DOWN = down
        self.DISTANCE = distance
        self.N = self.X = 0
        self.P = [None, None, None]  # Probability with confidence interval
        self.SMOOTHED = None

    def binom(self):
        if self.X > 0:
            self.P[1] = self.X / self.N
            self.P[0] = Functions.BinomLow(self.X, self.N, Globals.CONFIDENCE)
            self.P[2] = Functions.BinomHigh(self.X, self.N, Globals.CONFIDENCE)


def Array_Declaration():
    global P1D_ARRAY
    global P1D_GOAL_ARRAY
    P1D_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    P1D_GOAL_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    return None

def P1D_calculate():
    '''
    Calculate P(1D) for  all values of D&D
    '''
    print("calculating P(1D)", Functions.timestamp())
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT:
                if play.ODK == "OD":
                    if play.DISTANCE == play.YDLINE:
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT
                    else:
                        P1D_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT

    for down in P1D_ARRAY:
        for distance in down:
            distance.binom()
    for down in P1D_GOAL_ARRAY:
        for distance in down:
            distance.binom()

'''
   comment this function
   We're making all the P1D graphs, and doing it in a more organized way with
   one-liner filters. The way we had before with all the iterative loops was a
   nightmare. Eventually we need to include the plot figures themselves. We
   also need a function
'''


def P1D_PLOTS():
    '''
    Make the graphs for P(1D) for all downs
    '''
    # TODO: This is a horrifying mess, please clean up.
    P1D_goal_plot = [[], [], [], []]
    P1D_formats = [['', '', '0th', ''],
                   ['b', 'o', '1st', Functions.linearFit,
                    ((-1, 0), (0, 1.5)),
                    Functions.fitLabels(Functions.linearFit)],
                   ['r', 's', '2nd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)],
                   ['y', '^', '3rd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)]]


    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        error = numpy.array([[(x.P[2] - x.P[1]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD],
                             [(x.P[1] - x.P[0]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD]])
        fit = scipy.optimize.curve_fit(P1D_formats[down][3], xdata, ydata, bounds=P1D_formats[down][4])[0]
        r2 = Functions.RSquared(P1D_formats[down][3], fit, xdata, ydata)
        rmse = Functions.RMSE(P1D_formats[down][3], fit, xdata, ydata)
        plt.errorbar(xdata, ydata, yerr=error, ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1])
        plt.plot(numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1),
                 P1D_formats[down][3](numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1), *fit),
                 color=P1D_formats[down][0], label=P1D_formats[down][5].format(*fit, r2, rmse))
        plt.xlabel("Distance")
        plt.ylabel("P(1D)")
        plt.title(("P(1D) of " + Functions.ordinals[down] + " Down by Distance"))
        plt.grid(True)
        plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
        plt.legend(loc='best')
        plt.savefig(("Figures/P(1D)/P(1D) " + P1D_formats[down][2] + " Down"), dpi=1000)
        plt.show()

    #All downs?
    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1],
                     label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) all Downs"), dpi=1000)
    plt.show()

    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0],
                     fmt=P1D_formats[down][1], label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of & Goal for all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) &Goal"), dpi=1000)
    plt.show()
    return None

def teamseason():
    xdata = []
    ydata = []
    
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.OFFENSE == team and play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("$P(1D)_{1^{st}&10}$")
    plt.ylabel("$P(1D)_{2^{nd}&10}$")
    plt.title("$P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for defensive P(1D)
    '''
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.DEFENSE == team:
                            if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                                tempdata[play.DOWN][0] += 1
                                tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("Defensive $P(1D)_{1^{st}&10}$")
    plt.ylabel("Defensive $P(1D)_{2^{nd}&10}$")
    plt.title("Defensive $P(1D)_{1^{st}&10} vs. P(1D)_{2^{nd}&10}$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/Defensive P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for P(1D) by conference
    '''
    for season in range(2002, 2019):
        for conference in Globals.CISConferences:
            print(season, conference)
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and game.CONFERENCE == conference:
                    for play in game.playlist:
                        if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                try:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + conference + " logo.png", zoom=0.0075)
                except Exception:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/U SPORTS logo.png", zoom=0.0075)
    plt.xlabel("Conference $P(1D)_{1^{st}&10}$")
    plt.ylabel("Conference $P(1D)_{2^{nd} & 10}$")
    plt.title("Conference $P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.45, 0.7, 0.15, 0.5])
    plt.savefig(("Figures/P(1D)/Conference P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
>>>>>>> parent of 7093df1... Merge branch 'master' of https://github.com/christophermclement/U-Sports-Analytics
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 01:12:18 2018

@author: Chris Clement
"""
import Functions
import Globals
import matplotlib.pyplot as plt
import scipy.optimize
import numpy
import gc

P1D_ARRAY = []
P1D_GOAL_ARRAY = []


# This holds all the P1D data for a given D&D
class P1D():
    '''P1D objects hold information for a P1D state of down & distance
    '''

    def __init__(self, down, distance):
        self.DOWN = down
        self.DISTANCE = distance
        self.N = self.X = 0
        self.P = [None, None, None]  # Probability with confidence interval
        self.SMOOTHED = None

    def binom(self):
        if self.X > 0:
            self.P[1] = self.X / self.N
            self.P[0] = Functions.BinomLow(self.X, self.N, Globals.CONFIDENCE)
            self.P[2] = Functions.BinomHigh(self.X, self.N, Globals.CONFIDENCE)


def Array_Declaration():
    global P1D_ARRAY
    global P1D_GOAL_ARRAY
    P1D_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    P1D_GOAL_ARRAY = [[P1D(down, distance) for distance in range(Globals.DISTANCE_LIMIT)] for down in range(4)]
    return None

def P1D_calculate():
    '''
    Calculate P(1D) for  all values of D&D
    '''
    print("calculating P(1D)", Functions.timestamp())
    for game in Globals.gamelist:
        for play in game.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT:
                if play.ODK == "OD":
                    if play.DISTANCE == play.YDLINE:
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_GOAL_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT
                    else:
                        P1D_ARRAY[play.DOWN][play.DISTANCE].N += 1
                        P1D_ARRAY[play.DOWN][play.DISTANCE].X += play.P1D_INPUT

    for down in P1D_ARRAY:
        for distance in down:
            distance.binom()
    for down in P1D_GOAL_ARRAY:
        for distance in down:
            distance.binom()

'''
   comment this function
   We're making all the P1D graphs, and doing it in a more organized way with
   one-liner filters. The way we had before with all the iterative loops was a
   nightmare. Eventually we need to include the plot figures themselves. We
   also need a function
'''


def P1D_PLOTS():
    '''
    Make the graphs for P(1D) for all downs
    '''
    # TODO: This is a horrifying mess, please clean up.
    P1D_goal_plot = [[], [], [], []]
    P1D_formats = [['', '', '0th', ''],
                   ['b', 'o', '1st', Functions.linearFit,
                    ((-1, 0), (0, 1.5)),
                    Functions.fitLabels(Functions.linearFit)],
                   ['r', 's', '2nd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)],
                   ['y', '^', '3rd', Functions.exponentialDecayFit,
                    ((0, -numpy.inf, 0), (numpy.inf, 0, 0.5)),
                    Functions.fitLabels(Functions.exponentialDecayFit)]]


    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        error = numpy.array([[(x.P[2] - x.P[1]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD],
                             [(x.P[1] - x.P[0]) for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD]])
        fit = scipy.optimize.curve_fit(P1D_formats[down][3], xdata, ydata, bounds=P1D_formats[down][4])[0]
        r2 = Functions.RSquared(P1D_formats[down][3], fit, xdata, ydata)
        rmse = Functions.RMSE(P1D_formats[down][3], fit, xdata, ydata)
        plt.errorbar(xdata, ydata, yerr=error, ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1])
        plt.plot(numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1),
                 P1D_formats[down][3](numpy.arange(0, Globals.DISTANCE_LIMIT, 0.1), *fit),
                 color=P1D_formats[down][0], label=P1D_formats[down][5].format(*fit, r2, rmse))
        plt.xlabel("Distance")
        plt.ylabel("P(1D)")
        plt.title(("P(1D) of " + Functions.ordinals[down] + " Down by Distance"))
        plt.grid(True)
        plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
        plt.legend(loc='best')
        plt.savefig(("Figures/P(1D)/P(1D) " + P1D_formats[down][2] + " Down"), dpi=1000)
        plt.show()

    #All downs?
    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0], fmt=P1D_formats[down][1],
                     label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) all Downs"), dpi=1000)
    plt.show()

    for down in range(1, 4):
        xdata = numpy.array([x.DISTANCE for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        ydata = numpy.array([x.P[1] for x in P1D_GOAL_ARRAY[down] if x.N > Globals.THRESHOLD])
        plt.errorbar(xdata, ydata, linestyle="-", ms=3, color=P1D_formats[down][0],
                     fmt=P1D_formats[down][1], label=(Functions.ordinals(down) + " Down"))
    plt.xlabel("Distance")
    plt.ylabel("P(1D)")
    plt.title(("P(1D) of & Goal for all Downs by Distance"))
    plt.grid(True)
    plt.axis([0, Globals.DISTANCE_LIMIT + 1, 0, 1])
    plt.legend(loc='best')
    plt.savefig(("Figures/P(1D)/P(1D) &Goal"), dpi=1000)
    plt.show()
    return None

def teamseason():
    xdata = []
    ydata = []
    
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.OFFENSE == team and play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("$P(1D)_{1^{st}&10}$")
    plt.ylabel("$P(1D)_{2^{nd}&10}$")
    plt.title("$P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for defensive P(1D)
    '''
    for season in range(2002, 2019):
        for team in Globals.CISTeams:
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and (game.HOME == team or game.AWAY == team):
                    for play in game.playlist:
                        if play.DEFENSE == team:
                            if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                                tempdata[play.DOWN][0] += 1
                                tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + team + " logo.png", zoom=0.0075)
    plt.xlabel("Defensive $P(1D)_{1^{st}&10}$")
    plt.ylabel("Defensive $P(1D)_{2^{nd}&10}$")
    plt.title("Defensive $P(1D)_{1^{st}&10} vs. P(1D)_{2^{nd}&10}$")
    plt.grid()
    plt.axis([0.25, 0.75, 0.1, .6])
    plt.savefig(("Figures/P(1D)/Defensive P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
    gc.collect()
    '''
    Repeat the above graph but for P(1D) by conference
    '''
    for season in range(2002, 2019):
        for conference in Globals.CISConferences:
            print(season, conference)
            tempdata = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for game in Globals.gamelist:
                if game.game_date.year == season and game.CONFERENCE == conference:
                    for play in game.playlist:
                        if play.DISTANCE == 10 and play.P1D_INPUT is not None:
                            tempdata[play.DOWN][0] += 1
                            tempdata[play.DOWN][1] += play.P1D_INPUT
            if tempdata[1][0] > Globals.THRESHOLD and tempdata[2][0] > Globals.THRESHOLD / 2:
                try:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/" + conference + " logo.png", zoom=0.0075)
                except Exception:
                    Functions.imscatter(tempdata[1][1] / tempdata[1][0], tempdata[2][1] / tempdata[2][0], "Logos/U SPORTS logo.png", zoom=0.0075)
    plt.xlabel("Conference $P(1D)_{1^{st}&10}$")
    plt.ylabel("Conference $P(1D)_{2^{nd} & 10}$")
    plt.title("Conference $P(1D) 1^{st}&10 vs. 2^{nd}&10$")
    plt.grid()
    plt.axis([0.45, 0.7, 0.15, 0.5])
    plt.savefig(("Figures/P(1D)/Conference P(1D) 1st vs 2nd"), dpi=1000)
    plt.show()
>>>>>>> parent of 7093df1... Merge branch 'master' of https://github.com/christophermclement/U-Sports-Analytics
    gc.collect()