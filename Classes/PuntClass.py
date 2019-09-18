# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:19:26 2018

@author: Chris Clement
"""
import scipy
import matplotlib.pyplot as plt
import numpy
import Globals
import Classes.EPClass as EPClass
import Functions



class Punt():
    '''
    This holds data for punts, it's defined by the field position of the
    punt, and holds everything abot punts from that yardline. Resultant EP,
    gross, net, and spread
    '''
    def __init__(self, ydline):
        self.YDLINE = ydline

        self.EP = numpy.full(3, numpy.nan)  # Average EP value of this punt
        self.EP_ARRAY = []  # List holding all the EP data
        self.EP_BOOTSTRAP = Globals.DummyArray  # Bootstrap of EP to determine CI

        self.gross = numpy.full(3, numpy.nan)
        self.gross_array = []
        self.gross_bootstrap = Globals.DummyArray

        self.net = numpy.full(3, numpy.nan)
        self.net_array = []
        self.net_bootstrap = Globals.DummyArray

        self.spread = numpy.full(3, numpy.nan)
        self.spread_array = []
        self.spread_bootstrap = Globals.DummyArray

    def boot(self):
        '''
        Does the bootstrapping of EP, gross, net, spread, and calculates the
        average and confidence intervals.
        '''
        if len(self.EP_ARRAY) > 10:
            self.ep_bootstrap = Functions.bootstrap(self.EP_ARRAY)
            self.EP =\
                numpy.array([self.ep_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.EP_ARRAY) / float(len(self.EP_ARRAY))),
                 self.ep_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.gross_array) > 10:
            self.gross_bootstrap = Functions.bootstrap(self.gross_array)
            self.gross =\
                numpy.array([self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.gross_array) / float(len(self.gross_array))),
                 self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.net_array) > 10:
            self.net_bootstrap = Functions.bootstrap(self.net_array)
            self.net =\
                numpy.array([self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.net_array) / float(len(self.net_array))),
                 self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])

        if len(self.spread_array) > 10:
            self.spread_bootstrap = Functions.bootstrap(self.spread_array)
            self.spread =\
                numpy.array([self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.spread_array) / float(len(self.spread_array))),
                 self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])
        return None

PUNT_ARRAY = [Punt(ydline) for ydline in range(110)]

def P_boot():
    '''
    Tells every punt object in the array to run its boot function
    '''
    print("Bootstrapping punts", Functions.timestamp())
    for ydline in PUNT_ARRAY:
        ydline.boot()
    return None


def P_EP():
    '''
    Populates the punt EP array
    '''
    try:
        for g, game in enumerate(Globals.gamelist):
            for p, play in enumerate(game.playlist):
                if play.ODK == "P" and not(play.score_play == "SAFETY" and play.score_play_is_off):
                    # Need to filter out intentional safeties
                    if play.score_play:
                        PUNT_ARRAY[play.YDLINE].EP_ARRAY.append(numpy.float(Globals.score_values[play.score_play][1] * (1 if play.score_play_is_off else -1)))
                    else:
                        for n, nextPlay in enumerate(game.playlist[p + 1:]):
                            PUNT_ARRAY[play.YDLINE].EP_ARRAY.append(numpy.float(EPClass.EP_ARRAY[nextPlay.DOWN][nextPlay.DISTANCE][nextPlay.YDLINE].EP[1] * (1 if nextPlay.defense_offense == play.defense_offense else -1)))
                            break
                    if play.puntGross:
                        PUNT_ARRAY[play.YDLINE].gross_array.append(play.puntGross)
                    if play.puntNet:
                        PUNT_ARRAY[play.YDLINE].net_array.append(play.puntNet)
                    if play.puntSpread:
                        PUNT_ARRAY[play.YDLINE].net_array.append(play.puntSpread)
    except Exception as err:
        print(play.MULE, play.playdesc)
        print(err)
    return None


def P_PLOTS():
    '''
    Puts out the plots for punts
    TODO: Gross v net, Gross v spread
    '''

    # EP vs ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD])
    ydata = numpy.array([ydline.EP[1] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD])
    error = numpy.array([[ydline.EP[1] - ydline.EP[0] for ydline in PUNT_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD],
                         [ydline.EP[2] - ydline.EP[1] for ydline in PUNT_ARRAY if len(ydline.EP_ARRAY) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, numpy.array(xdata), numpy.array(ydata))
    r2 = Functions.RSquared(func, fit, numpy.array(xdata), numpy.array(ydata))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, yerr=error, fmt='D', color='purple', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set_(xlabel="Yardline", ylabel="EP(P)")
    fig.suptitle("EP(P) by Yardline")
    ax.grid(True)
    ax.axis([30, 111, -3, 2])
    ax.legend(loc='best')
    fig.savefig("Figures/P/EP(P)", dpi=1000)
    plt.close('all')
    gc.collect()

    #Gross vs ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.gross[1] for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.gross[1] - ydline.gross[0]) for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD],
                         [(ydline.gross[2] - ydline.gross[1]) for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set(xlabel="Yardline", ylabel="Gross yardage")
    fig.suptitle("Gross Punt Yardage by Starting Yardline")
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis([30, 111, 0, 50])
    fig.savefig("Figures/P/Punt gross by starting yardline", dpi=1000)
    plt.close('all')
    gc.collect()

    #Net vs ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.net[1] for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.net[1] - ydline.net[0]) for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD],
                         [(ydline.net[2] - ydline.net[1]) for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set(xlabel="Yardline", ylabel="Net Yardage")
    fig.suptitle("Punt Net by Yardline")
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis([30, 111, 0, 50])
    fig.savefig("Figures/P/Punt Net by yardline", dpi=1000)
    plt.close('all')
    gc.collect()

    # Spread vs Ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.spread[1] for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.spread[1] - ydline.spread[0]) for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD],
                         [(ydline.spread[2] - ydline.spread[1]) for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set(xlabel="Yardline", ylabel="Spread Yardage")
    fig.suptitle("Punt Spread by Yardline")
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis([30, 111, 0, 30])
    fig.savefig("Figures/P/Spread by Ydline", dpi=1000)
    plt.close('all')
    gc.collect()

    # Gross and net on same graph
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.gross[1] for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    ax.plot(xdata, ydata, color='blue', label='Gross')
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.net[1] for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    ax.plot(xdata, ydata, color='red', label='Net')
    ax.set(xlabel="Yardline", ylabel="Yardage")
    fig.suptitle("Punt Gross and Net by Yardline")
    ax.legend(loc='best')
    ax.grid(True)
    ax.axis([30, 111, 0, 50])
    fig.savefig("Figures/P/Gross and Net by Ydline", dpi=1000)
    plt.close('all')
    gc.collect()

    # PUNT EPA by ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD])
    ydata = numpy.array([PUNT_ARRAY[ydline].EP[1] - EPClass.EP_ARRAY[ydline].EP[1] for ydline in range(110) if PUNT_ARRAY[ydline].N > Globals.THRESHOLD])
    
    error = numpy.array([[ydline.EP[1] - ydline.EP[0] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD],
                         [ydline.EP[2] - ydline.EP[1] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, numpy.array(xdata), numpy.array(ydata))
    r2 = Functions.RSquared(func, fit, numpy.array(xdata), numpy.array(ydata))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(xdata, ydata, yerr=error, fmt='D', color='purple', ms=3)
    ax.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    ax.set(xlabel="Yardline", ylabel="EP(P)")
    fig.suptitle("EP(P) by Yardline")
    ax.grid(True)
    ax.axis([30, 111, -3, 2])
    ax.legend(loc='best')
    fig.savefig("Figures/P/EP(P)", dpi=1000)
    plt.close('all')