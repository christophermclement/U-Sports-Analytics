# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:15:48 2018

@author: Chris Clement
"""
import Globals
import numpy
import Classes.EPClass as EPClass
import Functions


class KO():
    '''
    KO classes are our simplest class, they deal with kickoffs, that only occur
    at a few specific yardlines.
    '''

    def __init__(self, ydline):
        self.YDLINE = ydline
        self.EP = numpy.array([None, None, None], dtype='float')
        self.EP_ARRAY = []
        self.BOOTSTRAP = None

    def calculate(self):
        if len(self.EP_ARRAY):  # Obviously not calculating if there's nothing
            try:
                self.EP[1] = sum(self.EP_ARRAY) / len(self.EP_ARRAY)
            except Exception as err:
                print("KO calc ERROR", self.YDLINE, self.EP_ARRAY)
                print(err)

    def boot(self):
        if len(self.EP_ARRAY) > 10:
            self.BOOTSTRAP = numpy.sort(numpy.array([numpy.average(
                numpy.random.choice(self.EP_ARRAY, len(self.EP_ARRAY), replace=True))
                for _ in range(Globals.BOOTSTRAP_SIZE)], dtype='f4'))
            self.EP[2] = self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]
            self.EP[0] = self.BOOTSTRAP[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)]

    def wipe(self):
        '''
        This just resets the value of all the attributes. We need it when we're iterating
        '''
        self.EP = numpy.array([None, None, None], dtype='float')
        self.EP_ARRAY = []
        self.BOOTSTRAP = None


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
    # TODO: Would we be better served by having SCOREvals in a dict? I think we
    would but how much rewriting would it take?
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
                                KO_ARRAY[play.YDLINE].EP_ARRAY.append(EPClass.EP_ARRAY[nextPlay.DOWN][nextPlay.DISTANCE][nextPlay.YDLINE].EP[1] * (1 if nextPlay.OFFENSE == play.OFFENSE else -1))
                                break
    except Exception as err:
        print("KO_EP ERROR", play.MULE, play.playdesc)
        print(err)