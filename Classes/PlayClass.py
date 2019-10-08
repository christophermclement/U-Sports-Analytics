# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:37:50 2018

@author: Chris Clement
"""

import Globals  # We probably won't need this but just in case for now
import Classes.EPClass as EPClass
import numpy

class play():
    '''
    This object represents a single football play. No non-plays like timeouts,
    but it will include "NO PLAY" plays such as false starts. Any information
    relating to the play that is not general to the game should be held here.
    '''

    def __init__(self, row, hscore, ascore, away_home, offense_is_home, qtr, away_home_timeouts, clock, MULE):

        # Need to carry this information because it affects some of the parsing
        self.MULE = MULE

        # The different formats have different row structures
        if self.MULE == 1 or self.MULE == 3:
            self.DD = row[0]  # The first cell has down & distance
            self.SPOT = None  # This data format doesn't have a separate FPOS
            self.playdesc = row[1]  # play description is in the second cell
        elif self.MULE == 2:
            self.DD = row[1]  # Mule 2 is structured differently
            self.SPOT = row[2]
            self.playdesc = row[3]

        self.offense_is_home = offense_is_home
        self.defense_offense = away_home if self.offense_is_home else away_home[::-1]
        self.defense_offense_timeouts = away_home_timeouts if self.offense_is_home else away_home_timeouts[::-1]
        self.defense_offense_score = [ascore, hscore] if self.offense_is_home else [hscore, ascore]
        self.offense_lead = self.defense_offense_score[1] - self.defense_offense_score[0]
        self.CLOCK = clock  # If there's any clock info
        self.QUARTER = numpy.int(qtr)  # Carry over the qtr

        # Here are all the other attributes we figure out via functions
        self.DOWN = None
        self.DISTANCE = None
        self.RP = None
        self.P_RSLT = None
        self.FPOS = None
        self.YDLINE = None
        self.DEFENSE = None
        self.ODK = None
        self.offense_wins = None
        self.FG_RSLT = None
        self.GAIN = None
        self.TIME = None
        
        self.score_play = None
        self.score_play_is_off = None

        self.P1D_INPUT = None
        
        self.next_score = None
        self.next_score_is_off = None

        # TODO: Make this into a dictionary
        self.TACKLER_ONE = None
        self.TACKLER_TWO = None
        self.PASSER = None
        self.RECEIVER = None
        self.RUSHER = None
        self.KICKER = None
        self.INTERCEPTER = None
        self.RETURNER = None

        self.puntGross = None
        self.puntNet = None
        self.puntSpread = None
        
        self.KOGross = None
        self.KONet = None
        self.KOSpread = None

        self.raw_EP = None
        self.EP_regression_list = []
        self.EP_classification_list = []
        self.EP_classification_values = []

        self.raw_EPA = None
        self.EPA_regression_list = []
        self.EPA_classification_values = []
        
        self.WP_list = []
        self.WPA_list = []
        
        self.FG_regression_list = []
        self.FG_classification_list = []
        
        self.realTime = None  # This is the real time of day for the play
        self.METAR = None
        self.headwind = None
        self.crosswind = None

    def DOWN_FN(self):
        '''
        Determines the down of the play, which is pretty basic string
        interpretation, depending on which mule we're using.
        '''
        try:  # No errors should really come up here
            if self.MULE == 1:  # Mule 1 we have to parse it from the DD cell
                if "0th" in self.DD:
                    self.DOWN = numpy.int(0)
                elif "1st" in self.DD:
                    self.DOWN = numpy.int(1)
                elif "2nd" in self.DD:
                    self.DOWN = numpy.int(2)
                elif "3rd" in self.DD:
                    self.DOWN = numpy.int(3)
                else:
                    print("DOWN ERROR:", self.MULE, self.playdesc)
            elif self.MULE == 2:  # In format 2 it's always the first charactor of the DD column
                self.DOWN = numpy.int(self.DD[0])
            elif self.MULE == 3:  # In format three down is the third character
                self.DOWN = numpy.int(self.DD[2])
        # Will catch if there's a non-numeric raising an exception with int()
        except Exception:
            print("Down Error", self.MULE, self.playdesc)
        return None

    def DISTANCE_FN(self):
        '''
        Distance to gain is determined here, again basic string interpretation.
        '''
        try:
            if self.MULE == 1:
                self.DISTANCE = numpy.int(self.DD[8:10])
            elif self.MULE == 2:
                self.DISTANCE = numpy.int(self.DD[2:])
            elif self.MULE == 3:
                self.DISTANCE = numpy.int(self.DD[4:6])
            if self.DISTANCE <= 0 or self.DISTANCE is None:  # distance can't be 0 or negative
                print("DISTANCE ERROR:", self.MULE, self.playdesc)
        except Exception as err:
            print("DISTANCE ERROR:", self.MULE, self.playdesc)
            print(err)
        return None

    def RP_FN(self): 
        '''
        Determining if the play is a run or a pass. It can be neither as well.
        TODO: What if we changed this to "is_pass" as a boolean, then left "None" for when it's neither?
        '''
        # No error-check, phrases are in playdesc or not
        if any(x in self.playdesc for x in ["pass", "sack", "scramble"]):
            self.RP = "P"
        elif "rush" in self.playdesc:
            self.RP = "R"
        return None

    def P_RSLT_FN(self):
        '''
        Determining the result of a pass, such as a completion or incompletion
        TODO: Is there a way to numpy this variable?
        '''
        if self.RP == "P":  # Obviously only interested in pass plays
            if "incomplete" in self.playdesc:  # Looking for key strings
                self.P_RSLT = "I"
            elif "complete" in self.playdesc:
                self.P_RSLT = "C"
            elif "intercept" in self.playdesc:
                self.P_RSLT = "X"
            elif "sack" in self.playdesc:
                self.P_RSLT = "S"
            elif "scramble" in self.playdesc or ("pass" in self.playdesc and "rush" in self.playdesc):
                self.P_RSLT = "R"
            elif "FAILED" in self.playdesc:  # catching 2-pt conversions
                self.P_RSLT = "I"
            elif "GOOD" in self.playdesc or "good" in self.playdesc:
                self.P_RSLT = "C"
            else:  # If we don't have any of the key phrases something is wrong
                print("P RSLT ERROR", self.MULE, self.playdesc)
        return None

    def FPOS_FN(self):
        '''
        Determining field position, which requires string interpretation for
        the absolute value and again for the +/- value
        '''
        try:
            if self.MULE == 1:  # String interpretation for each data format
                self.FPOS = numpy.negative(int(self.DD[-2:]), dtype='int8') if self.DD[-5:-2] == self.defense_offense[1] else numpy.int(int(self.DD[-2:]))
            elif self.MULE == 2:
                self.FPOS = numpy.negative(int(self.SPOT[-2:]), dtype='int8') if self.SPOT[:3] == self.defense_offense[1] else numpy.int(int(self.SPOT[-2:]))
            elif self.MULE == 3:
                self.FPOS = numpy.negative(int(self.DD[-2:]), dtype='int8') if self.DD[0] == self.DD[-3] else numpy.int(int(self.DD[-2:]))
            if self.FPOS == -55:  # at midfield we define as positive
                self.FPOS = numpy.int(55)
        except Exception as err:  # Will catch any non-ints
            print("FPOS ERROR", self.MULE, self.DD, self.playdesc)
            print(err)
        return None

    def ODK_FN(self):
        '''
        Identify an O/D play, punt, kickoff, FG, or other
        Some of this ends up being process of elimination, and the order is
        fairly sensitive to that, we need to make sure that some plays are not
        included before we perform certain checks.
        TODO: Is there a way to numpy this?
        '''
        if self.DOWN == 3 and "SAFETY" in self.playdesc and self.YDLINE > 75:
            self.ODK = "P"  # need to account for intentional safeties
        # kick attempt is for PAT, need to put FG before P because of that
        # weird play where a FG is blocked and then they punt it and fumble it
        elif "field goal" in self.playdesc or "kick attempt" in self.playdesc:
            self.ODK = "FG"
        elif "punt" in self.playdesc:  # Looking for key phrases
            self.ODK = "P"
            if self.YDLINE < 20:  # flags punts too close to EZ to make sense
                print("PUNT ERROR", self.MULE, self.YDLINE, self.playdesc)
        elif "kickoff" in self.playdesc:
            self.ODK = "KO"
        # other runs or passes are OD, but it comes last because of fakes
        elif self.RP is not None:
            self.ODK = "OD"
        elif "PENALTY" in self.playdesc:
            self.ODK = "PEN"
        else:
            # This serves as a general catch-all for odd plays
            print("ODK ERROR", self.MULE, self.playdesc)
        return None

    def FG_RSLT_FN(self):  
        '''
        Identify field goals as good, rouge, or missed. We have a problem at
        the 5 yard line because missed PATs are never considered as rouges,
        even if they should be considered that way for averaging purposes.
        TODO: Is there a way to numpy this?
        '''
        # No error-catching because we have else for misses
        if self.ODK == "FG":
            if "GOOD" in self.playdesc:
                self.FG_RSLT = "GOOD"
            elif "ROUGE" in self.playdesc:
                self.FG_RSLT = "ROUGE"
            # There are several ways to note failed FGs so this is a catch-all
            else:
                self.FG_RSLT = "MISSED"
        return None

    def GAIN_FN(self):
        '''
        Gain of the play is tricky because they use "loss of" instead of a
        negative gain, which means more complicated string interpretation.
        '''
        try:
            if self.RP is not None:
                if self.P_RSLT == "I" or self.P_RSLT == "X":  # Incompletions
                    self.GAIN = numpy.int(0)
                elif "no gain" in self.playdesc:  # "no gain" instead of 0
                    self.GAIN = numpy.int(0)
                # Successful 2-pt conversions
                elif "GOOD" in self.playdesc and self.DOWN == 0:
                    self.GAIN = self.YDLINE
                elif "FAILED" in self.playdesc:
                    self.GAIN = numpy.int(0)
                # They use "loss" instead of -ve gain, so need to flip sign
                elif "loss" in self.playdesc:
                    # If somehow there's a loss of 100 yards
                    if self.playdesc[self.playdesc.find("yard")-4] == 1:
                        self.GAIN = numpy.int(self.playdesc[self.playdesc.find("yard")-4:self.playdesc.find("yard")-1])
                    else:
                        self.GAIN = numpy.int(self.playdesc[self.playdesc.find("yard")-3:self.playdesc.find("yard")-1])
                else:
                    # Handles gains of 100+ yards
                    if self.playdesc[self.playdesc.find("yard") - 4] == 1:
                        self.GAIN = numpy.int(self.playdesc[self.playdesc.find("yard")-4:self.playdesc.find("yard")])
                    else:
                        self.GAIN = numpy.int(self.playdesc[self.playdesc.find("yard")-3:self.playdesc.find("yard")])
        except Exception:  # Will catch any non-ints
            print("Gain Error", self.MULE, self.playdesc)
        return None

    def YDLINE_FN(self):
        '''
        Converting FPOS into YDLINE, which has no sign, it is simply yards from goal line.
        '''
        try:
            if self.FPOS > 0:  # Simple conversion from FPOS tro YDLINE
                self.YDLINE = numpy.int(self.FPOS)
            elif self.FPOS < 0:
                self.YDLINE = numpy.int(110 + self.FPOS)
            else:
                print("YDLINE ERROR", self.MULE, self.playdesc)
            if self.YDLINE <= 0 or self.YDLINE >= 110 or self.DISTANCE > self.YDLINE:  # For out of range errors
                print("YDLINE ERROR", self.MULE, self.playdesc)
        except Exception as err:
            print("YDLINE ERROR", self.MULE, self.playdesc)
            print(err)
        return None

    def TACKLER_FN(self):
        '''
        Identifying the tackler on a play
        TODO: The list of tacklers is a nightmare, and needs major cleanup to
        be more consistent with naming conventions. Since there are 1000+
        tacklers in the system I don't really know how to approach this
        Herculean task, other than starting at one end and trying to identify
        the common names that are in there multiple times with slight
        variations, and hope that gets the total list under control.
        '''
        if "(" in self.playdesc:
            TACKLERS = self.playdesc[
                self.playdesc.find("(") + 1 : self.playdesc.find(")")]
            if ";" in TACKLERS:
                self.TACKLER_ONE = TACKLERS[: TACKLERS.find(";")]
                self.TACKLER_TWO = TACKLERS[TACKLERS.find(";") + 1 :]
            else:
                self.TACKLER_ONE = TACKLERS
        return None

    def PASSER_FN(self):
        '''
        Similar to tackler, identifying who threw the pass
        TODO: Same as tackler, clean up the set of names
        '''
        if "pass " in self.playdesc and self.RP == "P":  # Only want passes
            if self.playdesc.find("pass ") == 0:  # For when passer isn't named
                # TODO: SHould this be changed to the offense team abbr?
                self.PASSER = "TEAM"
            else:
                self.PASSER = self.playdesc[:self.playdesc.find("pass ")]
                for x in range(3):
                    self.PASSER = self.PASSER[self.PASSER.find(",") + 1:] if "," in self.PASSER else self.PASSER
                    self.PASSER = self.PASSER[self.PASSER.find(":") + 3:] if ":" in self.PASSER else self.PASSER
                self.PASSER = self.PASSER.strip()
            # TODO: UNCOMMENT THESE LINES AND THOSE IN MASTER TO SEE THE LIST OF UNIQUE RECEIVERS FOR CLEANUP
            if self.PASSER not in Globals.passerList:
                Globals.passerList.append(self.PASSER)
        return None

    def RECEIVER_FN(self):
        '''
        Similar to tackler, we want to know the targeted receiver. But it's not
        always well-identified and it's much harder to say where the name ends
        and so on.
        TODO: Make this function work? It's going to be a nightmare of string
        comprehensions.
        '''
        if "complete to" in self.playdesc and self.RP == "P":
            self.RECEIVER = self.playdesc[self.playdesc.find("complete to") + 11:]
            # This will do for completions
            self.RECEIVER = self.RECEIVER[:self.RECEIVER.find(" for ")] if " for " in self.RECEIVER else self.RECEIVER
            # This will do for penalties
            self.RECEIVER = self.RECEIVER[:self.RECEIVER.find(",")] if "," in self.RECEIVER else self.RECEIVER
            # This will do for incompletions with defenders involved
            self.RECEIVER = self.RECEIVER[:self.RECEIVER.find("(")] if "(" in self.RECEIVER else self.RECEIVER
            # Cuts out penalties
            self.RECEIVER = self.RECEIVER[:self.RECEIVER.find(". PENALTY")] if ". PENALTY" in self.RECEIVER else self.RECEIVER
            # remove trailing periods
            self.RECEIVER = self.RECEIVER[:-1] if self.RECEIVER[-1] == "." else self.RECEIVER
            # This will do for incompletions without defenders involved
            self.RECEIVER = self.RECEIVER.strip()
            # TODO: UNCOMMENT THESE LINES AND THOSE IN MASTER TO SEE THE LIST OF UNIQUE RECEIVERS FOR CLEANUP
            if self.RECEIVER not in Globals.receiverList:
                Globals.receiverList.append(self.RECEIVER)
        return None

    def RETURNER_FN(self):
        '''
        like tackler et al we want to figure out who returned the kick
        '''
        if self.ODK in ["P", "FG", "KO"]:
            if "return" in self.playdesc:
                self.RETURNER = self.playdesc[:self.playdesc.find(" return")]  # cuts out everything after the word "return"
                self.RETURNER = self.RETURNER[self.RETURNER.find(",") + 1:] if "," in self.RETURNER else self.RETURNER
                self.RETURNER = self.RETURNER.strip()
                if self.RETURNER not in Globals.returnerList:
                    Globals.returnerList.append(self.RETURNER)
        return None

    def RUSHER_FN(self):
        if self.RP == "R":
            self.RUSHER = self.playdesc[:self.playdesc.find(" rush")]  # cuts out everything after "rush"
            self.RUSHER = self.playdesc[self.RUSHER.find(","):] if "," in self.RUSHER else self.RUSHER
            if self.RUSHER not in Globals.rusherList:
                Globals.rusherList.append(self.RUSHER)
        # TODO: Are there any documented scrambles??
        return None

    def KICKER_FN(self):
        if self.ODK in ["P", "FG", "KO"]:
            if self.ODK == "KO":
                self.KICKER = self.playdesc
                self.KICKER = self.KICKER[:self.KICKER.find("kickoff")]
                self.KICKER= self.KICKER[self.KICKER.find(",") + 1:] if "," in self.KICKER else self.KICKER
                self.KICKER = self.KICKER.strip()
            elif self.ODK == "P":
                self.KICKER = self.playdesc
                self.KICKER = self.KICKER[:self.KICKER.find("punt")]
                self.KICKER= self.KICKER[self.KICKER.find(",") + 1:] if "," in self.KICKER else self.KICKER
                self.KICKER = self.KICKER.strip()
            elif self.ODK == "FG":
                self.KICKER = self.playdesc
                self.KICKER = self.KICKER[:self.KICKER.find("kick")] if "kick" in self.KICKER else self.KICKER
                self.KICKER = self.KICKER[:self.KICKER.find("field goal")] if "kick" in self.KICKER else self.KICKER
                self.KICKER= self.KICKER[self.KICKER.find(",") + 1:] if "," in self.KICKER else self.KICKER
                self.KICKER = self.KICKER.strip()
            if self.KICKER not in Globals.kickerList:
                Globals.kickerList.append(self.KICKER)
        return None

    def INTERCEPTER_FN(self):
        if self.P_RSLT == "X":
            self.INTERCEPTER = self.playdesc
            self.INTERCEPTER = self.INTERCEPTER[self.INTERCEPTER.find("intercepted") + 15:]
            self.INTERCEPTER = self.playdesc[self.INTERCEPTER.find(","):] if "," in self.INTERCEPTER else self.INTERCEPTER
            if self.INTERCEPTER not in Globals.intercepterList:
                Globals.intercepterList.append(self.INTERCEPTER)


    def puntGross_FN(self):
        '''
        Determine the gross yardage of a punt
        '''
        try:
            if self.ODK == "P":
                if self.score_play != "SAFETY":
                    if self.RP == None:
                        if not any(x in self.playdesc for x in ["BLOCKED", "blocked", "fake", "Fake"]):
                            if self.score_play == "ROUGE" and self.score_play_is_off:
                                self.puntGross = self.YDLINE
                            elif "yard" in self.playdesc:  # Plays that simply don't have the word "yards" shouldn't throw an error
                                self.puntGross = self.playdesc[self.playdesc.find("punt") + 4:]
                                self.puntGross = self.puntGross[:self.puntGross.find("yard") - 1]
                                self.puntGross = numpy.int(self.puntGross.strip())
        except Exception as err:
            if "PENALTY" in self.playdesc:
                pass
            else:
                print("puntGross Error", self.MULE, self.playdesc, self.puntGross)
                print(err)
            self.puntGross = None
        return None

    def puntSpread_FN(self):
        '''
        Finds the difference between gross and net if both were successfully calculated.
        TODO: We could reduce this to a one-liner
        '''
        if self.puntNet and self.puntGross:
            self.puntSpread = self.puntGross - self.puntNet
        return None

    def KOGross_FN(self):
        '''
        Gets the gross gain of the kickoff using some string interpretation
        '''
        try:
            if self.ODK == "KO":
                if self.score_play == "ROUGE" and self.score_play_is_off:
                    self.KOGross = self.YDLINE
                elif "yard" in self.playdesc:  # Plays that simply don't have the word "yards" shouldn't throw an error
                    self.KOGross = self.playdesc[self.playdesc.find("kickoff") + 8:]
                    self.KOGross = self.KOGross[:self.KOGross.find("yard") - 1]
                    self.KOGross = numpy.int(self.KOGross.strip())
        except Exception as err:
            if "PENALTY" in self.playdesc:
                pass
            else:
                print("KOGross Error", self.MULE, self.playdesc, self.puntGross)
                print(err)
            self.KOGross = None
        return None

    def KOSpread_FN(self):
        '''
        Finds the difference between gross and net if both were successfully calculated.
        TODO: We could reduce this to a one-liner
        '''
        if self.KOGross and self.KONet:
            self.KOSpread = self.KOGross - self.KONet
        return None

    def EP_wipe(self):
        '''
        Wipes all the EP-related variables so we can simplify the pickling
        TODO: Clean this up with the new approach to EP models
        '''
        self.raw_EP = None
        self.EP_regression_list = []
        self.EP_classification_list = []
        return None

    def WP_wipe(self):
        '''
        Wipes the WP_list
        TODO: Clean this up with the new approach to modeling
        '''
        self.WP_list = []
        return None

    def FG_wipe(self):
        '''
        Wipes the FG model results
        TODO: Make sure that this is compatible with the way we do the modeling now
        '''
        self.FG_classification_list = []
        self.FG_regression_list = []
        return None