# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:15:48 2018

@author: Chris Clement
"""
import Globals
import numpy
import Classes.EPClass as EPClass
import Functions
import matplotlib.pyplot as plt
import scipy
import gc

class KO():
    '''
    KO classes are our simplest class, they deal with kickoffs, that only occur
    at a few specific yardlines.
    '''

    def __init__(self, ydline):
        self.YDLINE = ydline
        self.EP = numpy.full(3, numpy.nan)
        self.EP_ARRAY = []
        self.EP_bootstrap = None

        self.gross_array = []
        self.gross = numpy.full(3, numpy.nan)
        self.gross_bootstrap = Globals.DummyArray

        self.net_array = []
        self.net = numpy.full(3, numpy.nan)
        self.net_bootstrap = Globals.DummyArray

        self.spread_array = []
        self.spread = numpy.full(3, numpy.nan)
        self.spread_bootstrap = Globals.DummyArray
        
        return None

    def calculate(self):
        if len(self.EP_ARRAY):  # Obviously not calculating if there's nothing
            try:
                self.EP[1] = numpy.mean(self.EP_ARRAY)
            except Exception as err:
                print("KO calc ERROR", self.YDLINE, self.EP_ARRAY)
                print(err)
        return None

    def boot(self):
        if len(self.EP_ARRAY) > 10:
            self.EP_bootstrap = Functions.bootstrap(self.EP_ARRAY)
            self.EP = numpy.array(
                [self.EP_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                numpy.mean(self.EP_ARRAY),
                self.EP_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.gross_array) > 10:
            self.gross_bootstrap = Functions.bootstrap(self.gross_array)
            self.gross = numpy.array(
                [self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                numpy.mean(self.EP_ARRAY),
                self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.net_array) > 10:
            self.net_bootstrap = Functions.bootstrap(self.net_array)
            self.net = numpy.array(
                [self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                numpy.mean(self.EP_ARRAY),
                self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.spread_array) > 10:
            self.spread_bootstrap = Functions.bootstrap(self.spread_array)
            self.spread = numpy.array(
                [self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                numpy.mean(self.EP_ARRAY),
                self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])
            
        return None

    def wipe(self):
        '''
        This just resets the value of all the attributes. We need it when we're iterating
        '''
        self.EP = numpy.full(3, numpy.nan)
        self.EP_ARRAY = []
        self.EP_bootstrap = Globals.DummyArray


KO_ARRAY = [KO(yardline) for yardline in range(110)]    


def KO_wipe():
    for YDLINE in KO_ARRAY:
        YDLINE.wipe()


def KO_calculate():
    for YDLINE in KO_ARRAY:
        YDLINE.calculate()


def KO_boot():
    print("Bootstrapping KO", Functions.timestamp())
    for YDLINE in KO_ARRAY:
        YDLINE.boot()


def KO_EP():
    '''
    Here's where we calculate the value of a kickoff in EP. We can't just use
    the "next score," Because it's overweighted to good teams, so we have to
    hunt down the next play, and then look up it's raw EP value. Since the
    next play should always be 1st & 10 we know we'll always have a solid EP
    value.
    '''
    try:
        for g, game in enumerate(Globals.gamelist):  # This probably doesn't need enumerating
            for p, play in enumerate(game.playlist):
                if play.ODK == "KO":
                    if play.score_play:
                        KO_ARRAY[play.YDLINE].EP_ARRAY.append(Globals.score_values[play.score_play][1] * (1 if play.score_play_is_off else -1))
                    else:
                        for nextPlay in game.playlist[p + 1:]:
                            if nextPlay.ODK == "OD":
                                KO_ARRAY[play.YDLINE].EP_ARRAY.append(EPClass.EP_ARRAY[nextPlay.DOWN][nextPlay.DISTANCE][nextPlay.YDLINE].EP[1] * (1 if nextPlay.defense_offense == play.defense_offense else -1))
                                break
    except Exception as err:
        print("KO_EP ERROR", play.MULE, play.playdesc)
        print(err)
    return None


def KO_counts():
    '''
    Building the gross, net, and spread arrays
    '''
    try:
        for game in Globals.gamelist:  # Prob doesn't need enumerating
            for play in game.playlist:
                if play.KOGross:
                    KO_ARRAY[play.YDLINE].gross_array.append(play.KOGross)
                if play.KONet:
                    KO_ARRAY[play.YDLINE].net_array.append(play.KONet)
                if play.KOSpread:
                    KO_ARRAY[play.YDLINE].spread_array.append(play.KOSpread)
    except Exception as ett:
        print("KO counts ERROR", play.MULE, play.playdesc)
        print(err)
    return None


def KO_plots():
    '''
    Make some graphs for kickoffs
    '''
    print("Making KO graphs", Functions.timestamp())
    # Make the raw EP graph
    xdata = numpy.array([ydline.YDLINE for ydline in KO_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.EP[1] for ydline in KO_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD])
    error = numpy.array([numpy.subtract(xdata, [ydline.EP[0] for ydline in KO_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD]),
                         numpy.subract([ydline.EP[2] for ydline in KO_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD], xdata)])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, numpy.array(xdata), numpy.array(ydata))
    r2 = Functions.RSquared(func, fit, numpy.array(xdata), numpy.array(ydata))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, yerr=error, fmt='D', color='purple', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set(xlabel="Yardline", ylabel="EP(P)")
    fig.suptitle("EP(KO) by Yardline")
    ax.grid(True)
    ax.axis([30, 90, -3, 2])
    ax.legend(loc='best')
    fig.savefig("Figures/KO/EP(KO)", dpi=1000)
    plt.close('all')
    gc.collect()




