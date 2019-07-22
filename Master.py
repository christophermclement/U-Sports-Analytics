# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:39:08 2018

@author: Chris Clement
"""

import csv  # to bring in the MULE data from CSV
import pickle  # To save objects to avoid recalculating every time
import numpy
import Classes.GameClass
from Classes import P1DClass
from Classes import ThirdDownClass
from Classes import KOClass
from Classes import PuntClass
from Classes import FGClass
from Classes import EPClass
import Functions
import WP
import Globals
import sys
import itertools
import gc

def import_mule(csvmule, mule):
    '''
    Uses the csv library to bring in the 3 data csv mules
    '''
    print("    importing mule", mule)
    for row in csvmule:
        # looking for new games
        if "vs." in row or any(" vs. " in x for x in row):
            # add the old TEMP to the list of games
            Globals.gamelist.append(Classes.GameClass.game(row[0], mule))
        elif any(row):
            Globals.gamelist[-1].rowlist.append(row)           
    return None


def score_bootstrap():
    '''
    This function calls all the bootstrap functions for the score values
    '''
    print("Bootstrapping scores", Functions.timestamp())
    Globals.TDval_BOOTSTRAP =\
        numpy.sort(6 + KOClass.KO_ARRAY[65].BOOTSTRAP +
                   numpy.random.binomial(FGClass.FG_ARRAY[5].N,
                                         FGClass.FG_ARRAY[5].P_GOOD[1],
                                         Globals.BOOTSTRAP_SIZE) /
                   FGClass.FG_ARRAY[5].N)
    Globals.ROUGEval_BOOTSTRAP =\
        numpy.sort(1 - EPClass.EP_ARRAY[1][10][75].BOOTSTRAP)

    if EPClass.EP_ARRAY[1][10][75].EP[1] > (-1) * KOClass.KO_ARRAY[65].EP[1]:
        Globals.FGval_BOOTSTRAP = numpy.sort(3 - EPClass.EP_ARRAY[1][10][75].BOOTSTRAP)
    else:
        Globals.FGval_BOOTSTRAP = numpy.sort(3 + KOClass.KO_ARRAY[65].BOOTSTRAP)

    if EPClass.EP_ARRAY[1][10][75].EP[1] > (-1) * KOClass.KO_ARRAY[75].EP[1]:
        Globals.SAFETYval_BOOTSTRAP = numpy.sort(-2 - EPClass.EP_ARRAY[1][10][75].BOOTSTRAP)
    else:
        Globals.SAFETYval_BOOTSTRAP = numpy.sort(-2 + KOClass.KO_ARRAY[75].BOOTSTRAP)

    Globals.SCOREvals[3][2] = Globals.TDval_BOOTSTRAP[int(len(Globals.TDval_BOOTSTRAP) * (1 - Globals.CONFIDENCE))]
    Globals.SCOREvals[3][0] = Globals.TDval_BOOTSTRAP[int(len(Globals.TDval_BOOTSTRAP) * (Globals.CONFIDENCE) - 1)]
    Globals.SCOREvals[0][2] = Globals.FGval_BOOTSTRAP[int(len(Globals.FGval_BOOTSTRAP) * (1 - Globals.CONFIDENCE))]
    Globals.SCOREvals[0][0] = Globals.FGval_BOOTSTRAP[int(len(Globals.FGval_BOOTSTRAP) * (Globals.CONFIDENCE) - 1)]
    Globals.SCOREvals[1][2] = Globals.ROUGEval_BOOTSTRAP[int(len(Globals.ROUGEval_BOOTSTRAP) * (1 - Globals.CONFIDENCE))]
    Globals.SCOREvals[1][0] = Globals.ROUGEval_BOOTSTRAP[int(len(Globals.ROUGEval_BOOTSTRAP) * (Globals.CONFIDENCE) - 1)]
    Globals.SCOREvals[2][2] = Globals.SAFETYval_BOOTSTRAP[int(len(Globals.SAFETYval_BOOTSTRAP) * (1 - Globals.CONFIDENCE))]
    Globals.SCOREvals[2][0] = Globals.SAFETYval_BOOTSTRAP[int(len(Globals.SAFETYval_BOOTSTRAP) * (Globals.CONFIDENCE) - 1)]
    print("TD      {0:5.4f}     {1:5.4f}     {2:5.4f}".format(*Globals.SCOREvals[3]))
    print("FG      {0:5.4f}     {1:5.4f}     {2:5.4f}".format(*Globals.SCOREvals[0]))
    print("ROUGE   {0:5.4f}     {1:5.4f}     {2:5.4f}".format(*Globals.SCOREvals[1]))
    print("SAFETY  {0:5.4f}     {1:5.4f}     {2:5.4f}".format(*Globals.SCOREvals[2]))


def iterate_scores():
    '''
    Iterates to find score values by adjusting for continuing effects until we reach convergence
    '''
    print("Iterating to find score values", Functions.timestamp())
    #TODO: Consider moving precision to a numpy.float128 type to get stupid levels of precision. At that point it's just an academic exercise in learning how to use numpy
    PRECISION = numpy.float64(100.0)  # This is the measure of convergence of score values
    TEMP = numpy.float64(0.0)  # Just a dummy to allow us to determine the change in value
    EPClass.EP_calculate()  # Need initial values for EP
    while PRECISION:  # Can adjust precision as needed
        EPClass.EP_calculate()
        KOClass.KO_wipe()
        FGClass.FG_wipe()
        KOClass.KO_EP()
        FGClass.FG_EP()
        KOClass.KO_calculate()
        FGClass.FG_calculate()

        TEMP = Globals.score_values["TD"][1]  # Adjust TD value
        Globals.score_values["TD"][1] = 6 + FGClass.FG_ARRAY[5].probabilities["GOOD"][1] + KOClass.KO_ARRAY[65].EP[1]
        PRECISION = abs(TEMP - Globals.score_values["TD"][1]) / Globals.score_values["TD"][1]

        TEMP = Globals.score_values["ROUGE"][1]  # Adjust ROUGE value
        Globals.score_values["ROUGE"][1] = 1 - EPClass.EP_ARRAY[1][10][75].EP[1]
        PRECISION = abs(Globals.score_values["ROUGE"][1] - TEMP) / Globals.score_values["ROUGE"][1]\
            if abs(Globals.score_values["ROUGE"][1] - TEMP) / Globals.score_values["ROUGE"][1]\
            > PRECISION else PRECISION

        # Put in a print about whether to take the ball at the 35 or make them kick
        TEMP = Globals.score_values["FG"][1]  # Adjust FG value
        if EPClass.EP_ARRAY[1][10][75].EP[1] > (-1) * KOClass.KO_ARRAY[65].EP[1]:
            Globals.score_values["FG"][1] = 3 - EPClass.EP_ARRAY[1][10][75].EP[1]
        else:
            Globals.score_values["FG"][1] = 3 + KOClass.KO_ARRAY[65].EP[1]
        PRECISION = abs(TEMP - Globals.score_values["FG"][1]) / Globals.score_values["FG"][1]\
            if abs(TEMP - Globals.score_values["FG"][1]) / Globals.score_values["FG"][1]\
            > PRECISION else PRECISION

        TEMP = Globals.score_values["SAFETY"][1]  # Adjust SAFETY value
        if EPClass.EP_ARRAY[1][10][75].EP[1] > (-1) * KOClass.KO_ARRAY[75].EP[1]:
            Globals.score_values["SAFETY"][1] = -2 - EPClass.EP_ARRAY[1][10][75].EP[1]
        else:
            Globals.score_values["SAFETY"][1] = -2 + KOClass.KO_ARRAY[75].EP[1]
        PRECISION = abs(TEMP - Globals.score_values["SAFETY"][1]) / Globals.score_values["SAFETY"][1]\
            if abs(TEMP - Globals.score_values["SAFETY"][1]) / Globals.score_values["SAFETY"][1]\
            > PRECISION else PRECISION
        print(PRECISION)
    print([x[1] for x in Globals.score_values])
    return None


def parser():
    '''
    parses the games and plays
    '''
    print("Parsing games and plays", Functions.timestamp())
    for game in Globals.gamelist:
            game.game_calc()
            game.make_plays()
            for play in game.playlist:
                play.DOWN_FN()  # No dependencies
                play.FPOS_FN()  # No dependencies
                play.DISTANCE_FN()  # No dependencies
                play.YDLINE_FN()  # Dependent on FPOS
                play.RP_FN()  # No dependencies
                play.P_RSLT_FN()  # Dependent on RP
                play.ODK_FN()  # Dependent on down, ydline, RP
                play.FG_RSLT_FN()  # Dependent on ODK
                play.GAIN_FN()  # Dependent on RP
                play.TACKLER_FN()
                play.PASSER_FN()  # Dependent on RP
                play.RECEIVER_FN()  # Dependent on RP
            game.OffIsHome_FN()  # No dependencies
            game.DEFENSE_FN()  # No dependencies
            game.O_D_SCORE_FN()  # No dependencies
            game.O_D_TO_FN()  # No dependencies
            game.O_WIN_FN()  # No dependencies
            game.TIME_FN()  # No dependencies
            game.SCORING_PLAY_FN()
            game.P1D_INPUT_FN()  # Dependent on Down, Distance, FPOS, Defense
            game.EP_INPUT_FN()  # Dependent on scoring play, Defense, O/D Score
            game.realTime_FN()  # Dependent on quarter
            game.METARList_FN()  # Dependent on real time
            game.playMETAR_FN()  # dependent on METAR list, real time
            game.head_cross_wind_FN()  # Dependent on METAR, playMETAR
            game.puntNet_FN()  # Dependent on ODK, YDLINE
            game.KONet_FN()  # Dependent on ODK, YDLINE
            # These have to come after because they rely on SCORING_PLAY
            for play in game.playlist:
                play.KOGross_FN()  # Dependent on ODK
                play.KOSpread_FN()  # Dependent on ODK, KOGross, KONet
                play.puntGross_FN()  # Dependent on ODK
                play.puntSpread_FN()  # Dependent on ODK, puntNet, puntGross


def reparse():
    if REPARSE_DATA:
        print("importing data", Functions.timestamp())
        with open("Data/CIS MULE 01.csv") as csvfile:
            import_mule(csv.reader(csvfile), 1)
        with open("Data/CIS MULE 02.csv") as csvfile:
            import_mule(csv.reader(csvfile), 2)
        with open("Data/CIS MULE 03.csv") as csvfile:
            import_mule(csv.reader(csvfile), 3)
        parser()
        P1DClass.P1D_calculate()
        EPClass.EP_COUNT()
        iterate_scores()
        print("Done iterating, pickling", Functions.timestamp())
        with open("Pickle/gamelist", 'wb') as file:
            pickle.dump(Globals.gamelist, file)
        with open("Pickle/SCOREvals", 'wb') as file:
            pickle.dump(Globals.SCOREvals, file)
        with open("Pickle/P1D_ARRAY", 'wb') as file:
            pickle.dump(P1DClass.P1D_ARRAY, file)
        with open("Pickle/P1D_GOAL_ARRAY", 'wb') as file:
            pickle.dump(P1DClass.P1D_GOAL_ARRAY, file)
        with open("Pickle/FG_ARRAY", 'wb') as file:
            pickle.dump(FGClass.FG_ARRAY, file)
        with open("Pickle/PUNT_ARRAY", 'wb') as file:
            pickle.dump(PuntClass.PUNT_ARRAY, file)
        with open("Pickle/KO_ARRAY", 'wb') as file:
            pickle.dump(KOClass.KO_ARRAY, file)
        with open("Pickle/EP_ARRAY", 'wb') as file:
            pickle.dump(EPClass.EP_ARRAY, file)
    else:
        print("Reusing pickled parsed data", Functions.timestamp())
        with open("Pickle/gamelist", 'rb') as file:
            Globals.gamelist = pickle.load(file)
        with open("Pickle/P1D_ARRAY", 'rb') as file:
            P1DClass.P1D_ARRAY = pickle.load(file)
        with open("Pickle/P1D_GOAL_ARRAY", 'rb') as file:
            P1DClass.P1D_GOAL_ARRAY = pickle.load(file)
        with open("Pickle/FG_ARRAY", 'rb') as file:
            FGClass.FG_ARRAY = pickle.load(file)
        with open("Pickle/KO_ARRAY", 'rb') as file:
            KOClass.KO_ARRAY = pickle.load(file)
        with open("Pickle/PUNT_ARRAY", 'rb') as file:
            PuntClass.PUNT_ARRAY = pickle.load(file)
        with open("Pickle/SCOREvals", 'rb') as file:
            Globals.SCOREvals = pickle.load(file)
        with open("Pickle/EP_ARRAY", 'rb') as file:
            EPClass.EP_ARRAY = pickle.load(file)
    return None


def recalc_ep():
    if RECALCULATE_EP:

        '''
        Bringing in these pickles and then retraining lets us jump to the warm start part of the game
        '''
        for m, model in enumerate(EPClass.EP_models):
            try:
                with open("Pickle/EPMODELS" + type(model).__name__, 'rb') as file:
                    model = pickle.load(file)
            except Exception:  # Basically, don't sweat it if you don't find one.
                pass

        EPClass.EP_Models()
        EPClass.BOOTSTRAP()
        PuntClass.P_EP()
        PuntClass.P_boot()
        KOClass.KO_boot()
        FGClass.FG_boot()
    
        score_bootstrap()

        print("    pickling", Functions.timestamp())
        with open("Pickle/EPARRAY", 'wb') as file:
            pickle.dump(EPClass.EP_ARRAY, file)
        for m, model in enumerate(EPClass.EP_models):  # Need to pickle models individually bc it can't pickle >4GB
            with open("Pickle/EPMODELS" + type(model).__name__, 'wb') as file:
                pickle.dump(model, file)
        with open("Pickle/gamelist", 'wb') as file:
            pickle.dump(Globals.gamelist, file)
        with open("Pickle/SCOREvals", 'wb') as file:
            pickle.dump(Globals.SCOREvals, file)
        with open("Pickle/PUNT_ARRAY", 'wb') as file:
            pickle.dump(PuntClass.PUNT_ARRAY, file)
        with open("Pickle/FG_ARRAY", 'wb') as file:
            pickle.dump(FGClass.FG_ARRAY, file)
        with open("Pickle/PUNT_ARRAY", 'wb') as file:
            pickle.dump(PuntClass.PUNT_ARRAY, file)
        with open("Pickle/KO_ARRAY", 'wb') as file:
            pickle.dump(KOClass.KO_ARRAY, file)
    else:  # Here we just load the pickled versions instead of recalculating
        print("Reusing pickled EP", Functions.timestamp())
        with open("Pickle/EPARRAY", 'rb') as file:
            EPClass.EP_ARRAY = pickle.load(file)
        for m, model in enumerate(EPClass.EP_models):
            with open("Pickle/EPMODELS" + type(model).__name__, 'rb') as file:
                EPClass.EP_models[m] = pickle.load(file)
        with open("Pickle/FG_ARRAY", 'rb') as file:
            FGClass.FG_ARRAY = pickle.load(file)
        with open("Pickle/KO_ARRAY", 'rb') as file:
            KOClass.KO_ARRAY = pickle.load(file)
        with open("Pickle/PUNT_ARRAY", 'rb') as file:
            PuntClass.PUNT_ARRAY = pickle.load(file)
        with open("Pickle/SCOREvals", 'rb') as file:
            Globals.SCOREvals = pickle.load(file)
    return None


def recalc_wp():
    gc.collect()
    if RECALCULATE_WP:
        for model in WP.WP_models:  # Speeds up the learning by giving a warm start
            try:  # If it doesn't work no biggie'
                with open("Pickle/WPMODELS" + type(model).__name__, 'rb') as file:
                    model = pickle.load(file)
                    print(type(model).__name__, "loaded")
            except Exception:
                pass
        WP.WP_Models()
        print("    pickling", Functions.timestamp())
        gc.collect()
        for model in WP.WP_models:  # Need to pickle models individually bc it can't pickle >4GB
            with open("Pickle/WPMODELS" + type(model).__name__, 'wb') as file:
                pickle.dump(model, file)
        with open("Pickle/gamelist", 'wb') as file:
            pickle.dump(Globals.gamelist, file)
    else:
        gc.collect()
        print("reusing pickled WP", Functions.timestamp())
        for model in WP.WP_models:
            with open("Pickle/WPMODELS" + type(model).__name__, 'rb') as file:
                model = pickle.load(file)
    return None


def recalc_fg():
    gc.collect()
    if RECALCULATE_FG:
        FGClass.FG_Models()
        print("Pickling FG models")
        with open("Pickle/FGMODELS", 'wb') as file:
            pickle.dump(FGClass.FG_models, file)
        with open("Pickle/gamelist", 'wb') as file:
            pickle.dump(Globals.gamelist, file)
    else:
        print("reusing pickled FG", Functions.timestamp())
        with open("Pickle/FGMODELS", 'rb') as file:
            FGClass.FG_models = pickle.load(file)
    return None


def redraw_plots():
    print("drawing plots")
    if DRAW_PLOTS:
        #PuntClass.P_PLOTS()
        #P1DClass.P1D_PLOTS()
        #P1DClass.teamseason()
        #FGClass.FG_PLOTS()
        #FGClass.FG_correlation()
        #EPClass.EP_correlation()
        #EPClass.EP_PLOTS()
        #EPClass.teamseason()
        WP.WP_correlation()
        #WP.WP_PLOTS()
        pass
    return None


REPARSE_DATA = True
RECALCULATE_EP = False
RECALCULATE_WP = True
RECALCULATE_FG = True
DRAW_PLOTS = True

'''
TODO: Make these one-liners in __main__
'''
KOClass.Array_Declaration()
PuntClass.array_declaration()

reparse()
EPClass.EP_regression()
recalc_ep()
recalc_wp()
recalc_fg()
redraw_plots()


'''
print (sorted(Globals.passerList), len(Globals.passerList))
print (sorted(Globals.receiverList), len(Globals.receiverList))
print (sorted(Globals.tacklerList), len(Globals.tacklerList))
# TODO: Clean up and render consistent the names used for players.
The receiver column also needs some improved parsing to get rid of some
extraneous crap. Un-comment these lines and those in the functions to see
the list of all unique passers and receivers. The list of TACKLERS probably
also needs cleaning
'''
    
print("ALL DONE", Functions.timestamp())

'''
# TODO: Convert all references to Globals.SCOREvals to the new dict, and fix up the tricky shit.

# TODO: Consider changing all float64 to longdouble, because why the fuck not

# TODO: rename the stuff like EP_models to make more sense, and create the parallel EP_classification and EP_regression, do the same for FG

# TODO: What if we changed a lot of this shit to numpy variables? And we can do the same in all the objects and really streamline the whole thing.
Would we see speed improvements? Maybe on som of the more complex shit, but I think it all gets fed around as numpy anyway in the background, or at least as C code.

# TODO: Convert this to Git instead of Google Drive for safekeeping

# TODO: Adding "return None" to the end of functions allows them to be collapsed in VS

# TODO: Add a kick returner function like passer, receiver, tackler

# TODO: Add a rusher function like passer, receiver, tackler

# TODO: Sort out everything related to third down and the big ol calculator

# TODO: Make the following functions: SEREND, ENDSER, SERBEG, BEGSER

# TODO: Create a standard set for all the colours/symbols we want to use in
matplotlib. ex:
    1st Down P(1D) is blue with circles,
    2nd down is red with squares,
    3rd down is yellow up triangles,
    1st down EP is green "x"es,
    P(FG) is now orange down triangles,
    EP(FG) brown stars
    EP(P) purple diamonds
    WP needs a colour

If it were in a usable list maybe we could even reference it directly? I feel
like a dictionary is the proper data structure to use but I don't know
if this really justifies it, since the calls will basically
all have to be manual anyway we might as well just have the list on paper or in
a note somewhere.

# TODO: Decide if it's worthwhile to switch our graphs to OOP instead of the
current state-based approach. We have a decent handle on the state method, but
it's awkward and unintuitive, every time we try to use a new feature it's a
slog. But it also is a huge hassle. Still, the learning curve will be steep
because the OOP method seems equally painful, but I think it will be worthwhile
long-term.

# TODO: There doesn't seem to be enough 2-pt conversion atTEMPts, so look into
the ODK function, it's probably logging a lot of field goals as 2-pt atTEMPts.

# TODO: Add docstrings for every method and function and generally improve the
commenting

# TODO: Why do I have a bunch of 0-5 at the 45???, 0-10 at the 5?
We need checks for when dist>Ydline
'''
    