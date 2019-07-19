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
PUNT_ARRAY = []


class Punt():
    '''
    This holds data for punts, it's defined by the field position of the
    punt, and holds everything abot punts from that yardline. Resultant EP,
    gross, net, and spread
    '''
    def __init__(self, ydline):
        self.N = 0  # Number of punts from this ydline
        self.YDLINE = ydline

        self.EP = [None, None, None]  # Average EP value of this punt
        self.ep_array = []  # List holding all the EP data
        self.ep_bootstrap = Globals.DummyArray  # Bootstrap of EP to determine CI

        self.gross = [None, None, None]
        self.gross_array = []
        self.gross_bootstrap = Globals.DummyArray

        self.net = [None, None, None]
        self.net_array = []
        self.net_bootstrap = Globals.DummyArray

        self.spread = [None, None, None]
        self.spread_array = []
        self.spread_bootstrap = Globals.DummyArray

    def boot(self):
        '''
        Does the bootstrapping of EP, gross, net, spread, and calculates the
        average and confidence intervals.
        '''
        if self.N > 10:
            self.ep_bootstrap = numpy.sort(numpy.array([numpy.average(numpy.random.choice(self.ep_array, self.N, replace=True))
                                                        for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))
            self.EP =\
                [self.ep_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.ep_array) / float(len(self.ep_array))),
                 self.ep_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]]

        if len(self.gross_array) > 10:
            self.gross_bootstrap = numpy.sort(numpy.array([numpy.average(numpy.random.choice(self.gross_array, len(self.gross_array), replace=True))
                                                           for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))

            self.gross =\
                [self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.gross_array) / float(len(self.gross_array))),
                 self.gross_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]]

        if len(self.net_array) > 10:
            self.net_bootstrap = numpy.sort(numpy.array([numpy.average(numpy.random.choice(self.net_array, len(self.net_array), replace=True))
                                                         for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))

            self.net =\
                [self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.net_array) / float(len(self.net_array))),
                 self.net_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]]

        if len(self.spread_array) > 10:
            self.spread_bootstrap = numpy.sort(numpy.array([numpy.average(numpy.random.choice(self.spread_array, len(self.spread_array), replace=True))
                                                            for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))

            self.spread =\
                [self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                 (sum(self.spread_array) / float(len(self.spread_array))),
                 self.spread_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]]
        return None


def array_declaration():
    '''
    Creates the punt array, which is technically a list, but it holds a punt object for every yardline
    # TODO: What if we just put this in main?
    '''
    global PUNT_ARRAY
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
    global PUNT_ARRAY
    try:
        for g, game in enumerate(Globals.gamelist):
            for p, play in enumerate(game.playlist):
                if play.ODK == "P":
                    # Need to filter out intentional safeties
                    if play.SCORING_PLAY != "O-SAFETY":
                        if play.puntGross:
                            PUNT_ARRAY[play.YDLINE].gross_array.append(play.puntGross)
                        if play.puntNet:
                            PUNT_ARRAY[play.YDLINE].net_array.append(play.puntNet)
                        if play.puntSpread:
                            PUNT_ARRAY[play.YDLINE].spread_array.append(play.puntSpread)
                        PUNT_ARRAY[play.YDLINE].N += 1
                        if play.SCORING_PLAY == "O-TD":
                            PUNT_ARRAY[play.YDLINE].ep_array.append(
                                Globals.SCOREvals[3][1])
                        elif play.SCORING_PLAY == "D-TD":
                            PUNT_ARRAY[play.YDLINE].ep_array.append(
                                Globals.SCOREvals[3][1] * (-1))
                        elif play.SCORING_PLAY == "O-ROUGE":
                            PUNT_ARRAY[play.YDLINE].ep_array.append(
                                Globals.SCOREvals[1][1])
                        elif play.SCORING_PLAY == "D-SAFETY":
                            PUNT_ARRAY[play.YDLINE].ep_array.append(
                                Globals.SCOREvals[2][1] * (-1))
                        else:
                            for n, nextPlay in enumerate(game.playlist[p + 1:]):
                                if nextPlay.ODK == "OD":
                                    if nextPlay.OFFENSE == play.OFFENSE:
                                        PUNT_ARRAY[play.YDLINE].ep_array.append(
                                            EPClass.EP_ARRAY[nextPlay.DOWN]
                                            [nextPlay.DISTANCE][nextPlay.YDLINE].EP[1])
                                        if EPClass.EP_ARRAY[nextPlay.DOWN]\
                                                [nextPlay.DISTANCE]\
                                                [nextPlay.YDLINE].EP[1] is None:
                                            print("PUNT ERROR", nextPlay.MULE, nextPlay.playdesc)
                                    elif nextPlay.OFFENSE == play.DEFENSE:
                                        PUNT_ARRAY[play.YDLINE].ep_array.\
                                            append(EPClass.EP_ARRAY[nextPlay.DOWN][nextPlay.DISTANCE]
                                                [nextPlay.YDLINE].EP[1] * (-1))
                                    else:  # error handling
                                        print("ODK ERROR:", play.playdesc, play.OFFENSE)
                                    break
                                # End of first half
                                elif nextPlay.QUARTER == 3 and game.playlist[n - 1] == 2:
                                    PUNT_ARRAY[play.YDLINE].ep_array.append(Globals.SCOREvals[4][1])
                                # end of second half into overtime
                                elif nextPlay.QUARTER == 5 and game.playlist[n-1] == 4:
                                    PUNT_ARRAY[play.YDLINE].ep_array.append(Globals.SCOREvals[4][1])
                            else:  # handles END OF GAME situation
                                PUNT_ARRAY[play.YDLINE].ep_array.append(Globals.SCOREvals[4][1])
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
    error = numpy.array([[ydline.EP[1] - ydline.EP[0] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD],
                         [ydline.EP[2] - ydline.EP[1] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, numpy.array(xdata), numpy.array(ydata))
    r2 = Functions.RSquared(func, fit, numpy.array(xdata), numpy.array(ydata))
    plt.errorbar(xdata, ydata, yerr=error, fmt='D', color='purple', ms=3)
    plt.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("EP(P)")
    plt.title("EP(P) by Yardline")
    plt.grid(True)
    plt.axis([30, 111, -3, 2])
    plt.legend(loc='best')
    plt.savefig("Figures/P/EP(P)", dpi=1000)
    plt.show()

    #Gross vs ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.gross[1] for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.gross[1] - ydline.gross[0]) for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD],
                         [(ydline.gross[2] - ydline.gross[1]) for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    plt.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    plt.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("Gross yardage")
    plt.title("Gross Punt Yardage by Starting Yardline")
    plt.legend()
    plt.grid(True)
    plt.axis([30, 111, 0, 50])
    plt.savefig("Figures/P/Punt gross by starting yardline", dpi=1000)
    plt.show()

    #Net vs ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.net[1] for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.net[1] - ydline.net[0]) for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD],
                         [(ydline.net[2] - ydline.net[1]) for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    plt.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    plt.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("Net Yardage")
    plt.title("Punt Net by Yardline")
    plt.legend()
    plt.grid(True)
    plt.axis([30, 111, 0, 50])
    plt.savefig("Figures/P/Punt Net by yardline", dpi=1000)
    plt.show()

    # Spread vs Ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.spread[1] for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD])
    error = numpy.array([[(ydline.spread[1] - ydline.spread[0]) for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD],
                         [(ydline.spread[2] - ydline.spread[1]) for ydline in PUNT_ARRAY if len(ydline.spread_array) > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, xdata, ydata)
    r2 = Functions.RSquared(func, fit, xdata, ydata)
    plt.errorbar(xdata, ydata, error, color='purple', fmt='D', ms=3)
    plt.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("Spread Yardage")
    plt.title("Punt Spread by Yardline")
    plt.legend()
    plt.grid(True)
    plt.axis([30, 111, 0, 30])
    plt.savefig("Figures/P/Spread by Ydline", dpi=1000)
    plt.show()

    # Gross and net on same graph
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.gross[1] for ydline in PUNT_ARRAY if len(ydline.gross_array) > Globals.THRESHOLD])
    plt.plot(xdata, ydata, color='blue', label='Gross')
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    ydata = numpy.array([ydline.net[1] for ydline in PUNT_ARRAY if len(ydline.net_array) > Globals.THRESHOLD])
    plt.plot(xdata, ydata, color='red', label='Net')
    plt.xlabel("Yardline")
    plt.ylabel("Yardage")
    plt.title("Punt Gross and Net by Yardline")
    plt.legend()
    plt.grid(True)
    plt.axis([30, 111, 0, 50])
    plt.savefig("Figures/P/Gross and Net by Ydline", dpi=1000)
    plt.show()

    # PUNT EPA by ydline
    xdata = numpy.array([ydline.YDLINE for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD])
    ydata = numpy.array([PUNT_ARRAY[ydline].EP[1] - EPClass.EP_ARRAY[ydline].EP[1] for ydline in range(110) if PUNT_ARRAY[ydline].N > Globals.THRESHOLD])
    
    error = numpy.array([[ydline.EP[1] - ydline.EP[0] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD],
                         [ydline.EP[2] - ydline.EP[1] for ydline in PUNT_ARRAY if ydline.N > Globals.THRESHOLD]])
    func = Functions.cubicFit
    fit = scipy.optimize.curve_fit(func, xdata, ydata)[0]
    rmse = Functions.RMSE(func, fit, numpy.array(xdata), numpy.array(ydata))
    r2 = Functions.RSquared(func, fit, numpy.array(xdata), numpy.array(ydata))
    plt.errorbar(xdata, ydata, yerr=error, fmt='D', color='purple', ms=3)
    plt.plot(numpy.arange(111), func(numpy.arange(111), *fit), color='purple', label=Functions.fitLabels(func).format(*fit, r2, rmse))
    plt.xlabel("Yardline")
    plt.ylabel("EP(P)")
    plt.title("EP(P) by Yardline")
    plt.grid(True)
    plt.axis([30, 111, -3, 2])
    plt.legend(loc='best')
    plt.savefig("Figures/P/EP(P)", dpi=1000)
    plt.show()