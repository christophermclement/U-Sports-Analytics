# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:37:40 2018

@author: Chris Clement
"""
import datetime  # to get the weekday from the date
import pytz  # Time zone info
from metar import Metar
import csv
import numpy
import Classes.PlayClass as PlayClass
import Classes.EPClass as EPClass
import Globals
import math
import traceback
playwords = ["rush", "pass", "sack", "kick", "punt", "field goal", "PENALTY"]

class game():
    '''
    Game objects represent games in the database. They hold all the data that
    is applicable to the entire game, like where it was held and on what day.
    The plays are held in "playlist," and this class has a number of methods
    that actually calculate attributes of the plays, because they require a
    lot more information about the plays around it, so it was easier to have
    the game send the information down to the plays than have the plays try 
    to find their own game and work from there.
    '''
    def __init__(self, statement, MULE):
        '''
        To create a game we just need the "vs." statement and we can draw all
        the critical information from there.
        '''
        self.rowlist = []  # Holds all the rows of raw data from the csv
        self.playlist = []  # Holds a list of all the plays as objects
        self.MULE = MULE  # Which data format is this from
        self.game_statement = statement  # The initial "vs. " statement
        # Grabs the Home and Away from the vs. statement
        self.away_home=[self.game_statement[0:3], self.game_statement[8:11]]
        self.H_WIN = None  # If the home team wins the game
        self.away_home_final = None  # Final score of the game
        self.LEAGUE = None  # Always "U Sports", but exists for future compatibility
        self.CONFERENCE = None  # The conference in which the game was played
        self.game_date = None  # A datetime object for when the game was played
        self.stadium = None  # What stadium the game was played, refers to a stadium object
        self.METARList = [] #  List of all the METARs during the game
        '''
        The season in which the game was played, always same as YEAR for
        Canadian football, as seasons don't span over New Year's Day, exists
        for compatibility with other leagues in the future
        '''
        self.SEASON = None

    def game_calc(self):
        '''
        Determines some of the basic info about the game - date, conference
        '''
        try:
            self.stadium = Globals.stadia[self.game_statement[28:]]
        except Exception:
            print("Stadium Error", self.game_statement)
        try:
            self.game_date = datetime.datetime(int(self.game_statement[12:16]),  # Year
                                               int(self.game_statement[17:19]),  # Month
                                               int(self.game_statement[20:22]),  # Day
                                               int(self.game_statement[23:25]),  # Hour
                                               int(self.game_statement[25:27]))  # Day
            # Fixes the time zone issue
            self.game_date = pytz.timezone(self.stadium.TZCode).localize(self.game_date)
        except Exception:
            print("game date error", self.game_statement)
        self.game_date = self.game_date.astimezone(pytz.utc)

        self.SEASON = self.game_date.year
        # Always U Sports, will get fancier if we include other leagues.
        self.LEAGUE = "U SPORTS"
        # These are to determine the conference of a game
        # These two conferences always have the same teams
        CWUAA = ["MAN", "SKH", "REG", "ALB", "CGY", "UBC", "SFU"]
        OUA = ["CAR", "OTT", "QUE", "TOR", "YRK", "MAC", "GUE", "WAT", "WLU", "WES", "WIN"]

        # Because Bishop's changed conferences in 2016 we need to adapt
        # Should this be a separate method?
        if self.game_date.year < 2016:
            RSEQ = ["BIS", "SHE", "MCG", "CON", "MON", "LAV"]
            AUS = ["SMU", "SFX", "MTA", "ACA"]
        else:
            RSEQ = ["SHE", "MCG", "CON", "MON", "LAV"]
            AUS = ["SMU", "SFX", "MTA", "ACA", "BIS"]
        # If both are in the same conf we assign that conf, otherwise non-con
        if all(team in CWUAA for team in self.away_home):
            self.CONFERENCE = "CWUAA"
        elif all(team in OUA for team in self.away_home):
            self.CONFERENCE = "OUA"
        elif all(team in RSEQ for team in self.away_home):
            self.CONFERENCE = "RSEQ"
        elif all(team in AUS for team in self.away_home):
            self.CONFERENCE = "AUS"
        else:
            self.CONFERENCE = "NONCON"
        return None

    def make_plays(self):
        '''
        Go through rowlist to create play objects, and get bookkeeping data
        Initialize quarter/score as 0, 2 timeouts per team, clock at 15:00.

        This gets a bit messy because we're handling a fair number of
        attributes but it actually does seem like the tidiest way to handle it.
        We could maybe refactor it but it would be even harder to read.
        # TODO: Refactor this into several sub-methods
        '''
        quarter = 1
        homescore = 0
        awayscore = 0
        away_home_timeouts = [2, 2]
        clock = "15:00"
        off = []  # No team is on offense until one is assigned

        # Looping thorugh the whole rowlist
        for row in self.rowlist:
            # Identifying new quarters and resetting the clock and timeouts
            if row[0] == "2nd":
                quarter = 2
                clock = "15:00"
            elif row[0] == "3rd":
                quarter = 3
                clock = "15:00"
                away_home_timeouts = [2, 2]
            elif row[0] == "4th":
                quarter = 4
                clock = "15:00"
            elif row[0] == "OT":
                quarter = 5
                clock = "00:00"

            # check for possession statements, but only in data formats 1 and 3
            if self.MULE == 1 or self.MULE == 3:
                if "drive start" in row[1]:
                    off = row[1][row[1].find("drive start") - 4:
                                 row[1].find("drive start") - 1]
            elif self.MULE == 2:
                if len(row[0]) == 3:
                    if row[0] not in ["1st", "2nd", "3rd", "4th"]:
                        off = row[0]
            if off not in self.away_home:
                print("POSSESSION ERROR", self.MULE, self.game_statement, row)

            try:  # Handling timeouts
                if self.MULE == 1 or self.MULE == 3:
                    if "TIMEOUT" in row[1]:  # Checking for timeout statements
                        TO = row[1].find("TIMEOUT")
                        TOTEAM = row[1][(TO + 8):(TO + 11)]
                        away_home_timeouts[self.away_home.index(TOTEAM)] -= 1
                elif self.MULE == 2:
                    if "TIMEOUT" in row[3]:
                        TOTEAM = row[3][row[3].index("TIMEOUT") + 8:row[3].index("TIMEOUT") + 11]
                        away_home_timeouts[self.away_home.index(TOTEAM)] -= 1
            except Exception as err:
                print("TIMEOUT ERROR", self.MULE, self.game_statement, row)
                print(err)

            try:
                if self.MULE == 1 or self.MULE == 3:  # find score statements
                    if len(row[1]) > 11 and len(row[1]) < 15:
                        if row[1][3] == " ":
                            if row[1][0:3] == self.away_home[0]:
                                awayscore = int(row[1][4:row[1].find(",")].
                                                lstrip(" "))
                                homescore = int(row[1][-2:].lstrip(" "))
                elif self.MULE == 2:
                    if len(row[3]) > 11 and len(row[3]) < 15:
                        if row[3][3] == " ":
                            if row[3][0:3] == self.away_home[0]:
                                awayscore = int(row[3][4:row[3].index(",")].
                                                lstrip(" "))
                                homescore = int(row[3][-2:].lstrip(" "))
            except Exception as err:
                print("SCORE ERROR", self.MULE, row, awayscore, homescore)
                print(err)

            # identify clock statemenrs
            try:
                if self.MULE == 1 or self.MULE == 3:
                    if ":" in row[1]:
                        clock = row[1][row[1].find(":") - 2:row[1].
                                       find(":") + 3]
                elif self.MULE == 2:
                    if ":" in row[3]:
                        clock = row[3][row[3].find(":") - 2:row[3].
                                       find(":") + 3]

                # identify plays
                # TODO: Why is this in the clock error try/exception???
                if self.MULE == 1:
                    if any(x in row[1] for x in playwords):
                        self.playlist.append(PlayClass.play(row, homescore, awayscore, self.away_home,
                                                            off == self.away_home[1],
                                                            quarter, away_home_timeouts, clock, self.MULE))
                        clock = None
                elif self.MULE == 2:
                    if any(x in row[3] for x in playwords):
                        self.playlist.append(PlayClass.play(row, homescore, awayscore, self.away_home,
                                                            off == self.away_home[1],
                                                            quarter, away_home_timeouts, clock, self.MULE))
                        clock = None
                elif self.MULE == 3:
                    if any(x in row[1] for x in playwords):
                        self.playlist.append(PlayClass.play(row, homescore, awayscore, self.away_home,
                                                            off == self.away_home[1],
                                                            quarter, away_home_timeouts, clock, self.MULE))
                        clock = None
            except Exception as err:
                print("CLOCK ERROR", self.MULE, row)
                print(err)

        self.away_home_final = [awayscore, homescore]
        self.home_wins = True if homescore > awayscore else False
        return None

    def O_WIN_FN(self):
        '''
        Returns True if the current offensive team wins
        '''
        for play in self.playlist:
            # Match offense to home or away and set the win accordingly
            play.offense_wins = not(play.offense_is_home ^ self.home_wins)
        return None

    def TIME_FN(self):
        '''
        Gets us the amount of time remaining in the game, in seconds, by splining between the known clock statements and quarter starts
        '''
        self.playlist[0].TIME = 3600  # Opening kickoff is at 15:00
        try:
            # First loop through converts the clock statements to time
            for play in self.playlist:
                if play.CLOCK is not None:
                    play.TIME = int(3600 - 900 * play.QUARTER + int(play.
                                    CLOCK[-2:]) + 60 * int(play.CLOCK[:2]))
                if play.QUARTER == 5:  # Handles OT
                    play.TIME = 0

            if self.playlist[-1].TIME is None:  # Last play of game
                self.playlist[-1].TIME = 0

            for p, play in enumerate(self.playlist):  # Spline gaps
                if play.TIME is None:  # Find start of gap
                    # Find end of gap
                    for n, nextPlay in enumerate(self.playlist[p:]):
                        if nextPlay.TIME is not None:  # If gap has ended
                            for s, spline in enumerate(self.playlist[p:p + n]):
                                spline.TIME = self.playlist[s - 1].TIME -\
                                    (self.playlist[p - 1].TIME -
                                     nextPlay.TIME) / (p)
                            p = n
                            break

            for play in self.playlist:  # Third loop to round
                play.TIME = int(round(play.TIME, 0))
        except Exception:  # A lot of misc errors can happen with all the loops
            print("TIME ERROR", self.MULE, p, play.playdesc)
        return None

    def SCORING_PLAY_FN(self):
        '''
        Identifies scoring plays and the team that scores them
        '''
        # Need to find all the plays with scores
        try:
            for p, play in enumerate(self.playlist):
                if "ROUGE" in play.playdesc:
                    play.score_play = "ROUGE"
                    play.score_play_is_off = True
                elif play.FG_RSLT == "GOOD" and play.DOWN:  # Filter PATs
                    play.score_play = "FG"
                    play.score_play_is_off = True
                elif "SAFETY" in play.playdesc:
                    play.score_play = "SAFETY"
                    if play.YDLINE < 65:
                        play.score_play_is_off = False
                    else:
                        play.score_play_is_off = True
                elif "TOUCHDOWN" in play.playdesc:
                    play.score_play = "TD"
                    for nextPlay in self.playlist[p + 1:]:
                        if any(x in nextPlay.playdesc for x in ["attempt", "kickoff"]): # Usu. when there's a score statement before PAT
                            play.score_play_is_off = True if nextPlay.defense_offense == play.defense_offense else False
                            break
                    else:
                        play.score_play_is_off = True if nextPlay.defense_offense_score[0] > play.defense_offense_score[0] else False
                elif play == self.playlist[-1]:  # end of game
                    play.score_play = "HALF"
                    play.score_play_is_off == False
                elif play.QUARTER == 2 and self.playlist[p+1].QUARTER == 3:  # halftime
                    play.score_play == "HALF"
                    play.score_play_is_off == False
                elif play.QUARTER == 4 and self.playlist[p+1].QUARTER == 5:  # OT
                    play.score_play = "HALF"
                    play.score_play_is_off == False
        except Exception as err:
            print("SCORING PLAY ERROR", self.MULE, self.playlist[p].playdesc)
            traceback.print_exc()
            print(err)
        return None

    def P1D_INPUT_FN(self):
        '''
        Returns True if the offense ends up converting a first down within the drive
        '''
        for play in self.playlist:
            if play.ODK == "OD":  # We only care about P(1D) for OD plays
                if play.DOWN > 0:  # We don't care about 2-pt conversions
                    # If the O scores a TD then it's obviously good
                    if play.score_play == "TD":
                        play.P1D_INPUT = play.score_play_is_off
                    elif play.next_score == "SAFETY" :  # Safeties are also bad, and a D-SAFETY certainly means some kind of change of possession
                            play.P1D_INPUT = False
                    else:
                        # Now we loop through the rest of the plays
                        for nextPlay in self.playlist[self.playlist.index(play) + 1:]:
                            # If there's a change of possession it's a fail
                            if nextPlay.defense_offense != play.defense_offense:
                                play.P1D_INPUT = False
                                break
                            # If offense scores a touchdown that's good
                            elif nextPlay.next_score == "TD":
                                play.P1D_INPUT = nextPlay.next_score_is_off
                                break
                                # If there's a non-OD play it implies a failure
                            elif nextPlay.next_score == "HALF":
                                play.P1D_INPUT = False
                                break
                            elif nextPlay.ODK in ["P", "FG", "KO"]:
                                play.P1D_INPUT = False
                                break
                            elif nextPlay == self.playlist[-1]:
                                play.P1D_INPUT = False
                                break
                            elif nextPlay.DOWN == 1 and (nextPlay.DISTANCE == 10 or nextPlay.DISTANCE == nextPlay.YDLINE):  # 1st Down
                                play.P1D_INPUT = True
                                break
                        else:
                            play.P1D_INPUT = False  # End of game, finding nothing
                elif play.DOWN == 0:  # Handling 2-point conversions
                    if "GOOD" in play.playdesc:
                        play.P1D_INPUT = True
                    else:
                        play.P1D_INPUT = False
        return None

    def EP_INPUT_FN(self):
        '''
        For each play gives the next type of score and whether the current offensive team will be the scoring team
        '''
        for p, play in enumerate(self.playlist):  # Loop through the playlist
            # Looping through all the plays going forward
            for nextPlay in self.playlist[p:]:
                if nextPlay.score_play:  # If there's a scoring play
                    # Need to match the scoring team with the current offense
                    play.next_score = nextPlay.score_play
                    if play.next_score == "HALF":
                        play.next_score_is_off = False
                    else:
                        play.next_score_is_off = not(play.offense_is_home ^ nextPlay.offense_is_home) * nextPlay.score_play_is_off
                    break
            else:
                print("EP INPUT ERROR", play.MULE, play.DOWN, play.DISTANCE, play.playdesc)
        return None

    def realTime_FN(self):
        '''
        Determines the actual time of day of each play, in Zulu
        Estimates the length of the game and splines along it.
        '''
        if self.playlist[-1].QUARTER == 4:
            for p, play in enumerate(self.playlist):
                if play.QUARTER < 3:
                    play.realTime = self.game_date\
                        + datetime.timedelta(hours=2, minutes=45)\
                        / len(self.playlist) * p
                else:  # Second half
                    play.realTime = self.game_date +\
                        datetime.timedelta(hours=2, minutes=45)\
                        / len(self.playlist) * p + datetime.timedelta(minutes=15)
        else:  # Handle OT games
            for p in range(len([x for x in self.playlist if x.QUARTER < 5])):
                if self.playlist[p].QUARTER < 3:
                    self.playlist[p].realTime = self.game_date\
                        + datetime.timedelta(hours=2, minutes=45)\
                        / len([x for x in self.playlist if x.QUARTER < 5]) * p
                else:  # Second half
                    self.playlist[p].realTime = self.game_date\
                        + datetime.timedelta(hours=2, minutes=45)\
                        / len([x for x in self.playlist if x.QUARTER < 5])\
                        * p + datetime.timedelta(minutes=15)
            # Handle OT
            for o in range(len([x for x in self.playlist if x.QUARTER == 5])):
                self.playlist[o + len([x for x in self.playlist if x.QUARTER < 5])].realTime = self.game_date + datetime.timedelta(hours=3) + datetime.timedelta(minutes=30) / len([x for x in self.playlist if x.QUARTER == 5]) * o
        return None

    def METARList_FN(self):
        '''
        Goes through the appropriate CSV to get the METARs appropriate to that game, accounting for the time zone and Zulu time
        We get killed by the strptime
        '''
        try:
            if self.stadium.isDome:
                startMETAR = self.stadium.airport + " "\
                    + str(self.game_date.day).zfill(2)\
                    + str(self.game_date.hour).zfill(2)\
                    + "00 00000KT 15SM CLR 20/16 A2992"
                endMETAR = self.stadium.airport + " "\
                    + str(self.game_date.day).zfill(2)\
                    + str(self.game_date.hour + 1).zfill(2)\
                    + "00 00000KT 15SM CLR 20/16 A2992"
                self.METARList.append(Metar.Metar(
                        startMETAR, month=self.game_date.month,
                        year=self.game_date.year))
                self.METARList.append(Metar.Metar(
                        endMETAR, month=self.game_date.month,
                        year=self.game_date.year))
            else:
                rowtime = datetime.datetime(1900, 1, 1).replace(tzinfo=pytz.utc)
                with open("METAR/" + self.stadium.airport + " " + str(self.game_date.year) + ".csv") as csvfile:
                    metarcsv = csv.reader(csvfile, delimiter=';')
                    for r, row in enumerate(metarcsv):
                        if len(row) > 1:
                            if len(row[1]) > 8:
                                if int(row[1][5:7]) >= self.game_date.month:
                                    if int(row[1][8:10]) >= self.game_date.day:
                                        rowtime = datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M").replace(tzinfo=pytz.utc)
                                        # this grabs anything within 1 hour of KO
                                        if rowtime > self.game_date - datetime.timedelta(hours=1):
                                            self.METARList.append(Metar.Metar(row[2], month=rowtime.month, year=rowtime.year,
                                                    utcdelta=datetime.timedelta()))
                                            self.METARList[-1].time = self.METARList[-1].time.replace(tzinfo=pytz.utc)  # Does the UTC Conversion on the metar object we just created
                        if rowtime >= self.playlist[-1].realTime:
                            break
                if self.METARList == []:
                    print("METAR List Empty Error", self.MULE,
                          self.game_statement, self.stadium.airport,
                          self.game_date.time(), self.playlist[-1].realTime)
                # If 2nd METAR is <= to the start of the game, delete 1st
                while True:
                    if self.METARList[1].time <= self.game_date:
                        self.METARList = self.METARList[1:]
                    else:
                        break  # Or we're ok and we shouldn't waste time
        except Exception as err:
            print("METAR List Error", self.MULE, self.game_statement,
                  self.stadium.airport, self.game_date.year)
            print(err)
        return None

    def playMETAR_FN(self):
        '''
        Goes through METAR list for the game and finds the one tha's time-stamped nearest to that game
        '''
        try:
            for play in self.playlist:
                for M, METAR in enumerate(self.METARList):
                    if METAR.time > play.realTime:
                        if abs(METAR.time - play.realTime) <\
                            abs(self.METARList[M - 1].time - play.realTime):
                            play.METAR = METAR
                        else:
                            play.METAR = self.METARList[M - 1]
                        break
        except Exception as err:
            print("METAR Error", self.MULE, play.playdesc, self.stadium.airport, str(self.game_date.year))
            print(err)
        return None

    def puntNet_FN(self):
        '''
        Calculates the net yardage of a punt play, just looking at the yardline of the current and next play
        '''
        try:
            for p, play in enumerate(self.playlist):
                if play.ODK == "P":
                    if not (play.score_play == "SAFETY" and play.score_play_is_off):
                        if play == self.playlist[-1]:
                            pass  # end of game
                        elif play.score_play == "HALF":
                            pass  # Halftime
                        elif play.score_play == "TD":
                            if play.score_play_is_off:
                                play.puntNet == play.YDLINE
                            else:
                                play.puntNet == 110 - play.YDLINE
                        elif self.playlist[p + 1].defense_offense == play.defense_offense:
                            play.puntNet = play.YDLINE - self.playlist[p + 1].YDLINE
                        else:
                            play.puntNet = play.YDLINE - (110 - self.playlist[p + 1].YDLINE)
        except Exception as err:
            print("puntNet Error", play.MULE, play.playdesc)
            print(err)
            play.puntNet = None
        return None

    def KONet_FN(self):
        '''
        Finds the net distance of the kickoff from the spot of the next play
        '''
        try:
            for p, play in enumerate(self.playlist):
                if play.ODK == "KO":
                    if play == self.playlist[-1]:
                        pass  # end of game
                    elif play == self.playlist[-1]:
                        pass  # end of game
                    elif play.score_play == "HALF":
                        pass  # Halftime
                    elif play.score_play == "TD":
                        if play.score_play_is_off:
                            play.KONet = play.YDLINE
                        else:
                            play.KONet = 110 - play.YDLINE
                    elif self.playlist[p + 1].defense_offense == play.defense_offense:
                        play.KONet = play.YDLINE - self.playlist[p + 1].YDLINE
                    else:
                        play.KONet = play.YDLINE - (110 - self.playlist[p + 1].YDLINE)
        except Exception as err:
            print("KONet Error", self.MULE, play.playdesc)
            print(err)
            play.KONet = None
        return None

    def EPA_FN(self):
        '''
        Gets the EPA of a play based on this play and the next, factoring in scoring plays and such.
        TODO: Rework with the new system for handling EP_INPUT
        TODO: Rework with the new system of EP classification/regression models
        '''
        for play in self.playlist:
            if play.DISTANCE < Globals.DISTANCE_LIMIT:
                play.raw_EP = EPClass.EP_ARRAY[play.DOWN][play.DISTANCE][play.YDLINE].EP
            else:
                play.raw_EP = EPClass.EP_ARRAY[play.DOWN][Globals.DISTANCE_LIMIT - 1][play.YDLINE].EP
        try:
            for p, play in enumerate(self.playlist):
                if play.score_play:
                    play.raw_EPA = Globals.score_values[play.score_play].EP[1] * (1 if play.score_play_is_off else -1) - play.raw_EP[1]
                    play.EPA_classification_values = numpy.subtract(Globals.score_values[play.score_play].EP[1] 
                                                                    if play.score_play_is_off 
                                                                    else numpy.negative(Globals.score_values[play.score_play].EP[1]),
                                                                    play.EP_classification_values)
                    play.EPA_regression_list = numpy.subtract(Globals.score_values[play.score_play].EP[1] * (1 if play.score_play_is_off else -1), play.EP_regression_list)
                else:  # Almost every other condition
                    play.raw_EPA = (self.playlist[p + 1].raw_EP[1] * (1 if play.defense_offense == self.playlist[p+1].defense_offense else -1) - play.raw_EP[1])
                    play.EPA_regression_list = numpy.subtract(self.playlist[p + 1].EP_regression_list 
                                                              if play.defense_offense == self.playlist[p+1].defense_offense 
                                                              else numpy.negative(self.playlist[p + 1].EP_regression_list), 
                                                              play.EP_regression_list)
                    play.EPA_classification_values = numpy.subtract(self.playlist[p + 1].EP_classification_values 
                                                                    if play.defense_offense == self.playlist[p+1].defense_offense 
                                                                    else numpy.negative(self.playlist[p + 1].EP_classification_values),
                                                                    play.EP_classification_values)
        except Exception as err:
            print("EPA Error", self.MULE, play.DOWN, play.DISTANCE, play.YDLINE, play.playdesc)
            print(err)
            traceback.print_exc()
        return None

    def WPA_FN(self):
        '''
        Gets the WPA betrween each play and the next
        TODO: Rework this into something tidier?
        '''
        try:
            for p, play in enumerate(self.playlist):
                if play == self.playlist[-1]:  # End of game
                    if play.offense_wins:
                        play.WPA_list = [1 - x for x in play.WP_list]
                    else:
                        play.WPA_list = [-x for x in play.WP_list]
                else:
                    if play.defense_offense == self.playlist[p + 1].defense_offense:
                        play.WPA_list = list(numpy.subtract(self.playlist[p + 1].WP_list, play.WP_list))
                    else:
                        play.WPA_list = list(1 - numpy.subtract(self.playlist[p + 1].WP_list, play.WP_list))
        except Exception as err:
            print("WPA Error", self.MULE, play.DOWN, play.DISTANCE, play.YDLINE, play.playdesc)
            print(err)
            traceback.print_exc()
        return None

    def head_cross_wind_FN(self):
        '''
        Determines the vector components of the wind from the METAR.
        Game level because it needs the game stadium
        '''
        
        for play in self.playlist:
            try:
                if play.METAR is None:  # If there's a missing METAR for some reason
                    print("No METAR", play.MULE, play.DOWN, play.DISTANCE, play.YDLINE, play.playdesc)
                elif play.METAR.wind_speed is not None:  # Often wind speed simply isn't measured
                    if play.METAR.wind_gust:
                        wind_speed = (play.METAR.wind_speed._value + play.METAR.wind_gust._value) / 2
                    else:
                        wind_speed = play.METAR.wind_speed._value

                    if play.METAR.wind_dir_from:  # Handles when wind is between two values
                        '''
                        TODO: Here's the thing, we actually want to find the max value within the range, not the larger of the two extremes. Figure it out.
                        '''
                        startangle = play.METAR.wind_dir_from._degrees
                        endangle = play.METAR.wind_dir_to._degrees
                        if startangle > endangle:
                            endangle += 360
                        anglerange = numpy.arange(startangle, endangle + 1)
                        play.headwind = numpy.amax(numpy.abs(numpy.cos(numpy.radians(self.stadium.orientation - anglerange)) * wind_speed))
                        play.crosswind = numpy.amax(numpy.abs(numpy.sin(numpy.radians(self.stadium.orientation - anglerange)) * wind_speed))
                    elif not play.METAR.wind_dir:  # Handles variable winds
                        play.headwind = wind_speed
                        play.crosswind = wind_speed
                    elif play.METAR.wind_speed._value == 0:  # Handles stupidity like VRB00 where it won't give us components bc it lacks direction
                        play.headwind = 0
                        play.crosswind = 0
                    elif play.METAR and play.METAR.wind_speed is not None and play.METAR.wind_dir is not None and play.METAR.temp is not None:  # Base case
                        play.headwind = abs(math.cos(math.radians(self.stadium.orientation - play.METAR.wind_dir._degrees)) * wind_speed)
                        play.crosswind = abs(math.sin(math.radians(self.stadium.orientation - play.METAR.wind_dir._degrees)) * wind_speed)
            except Exception as err:
                print("head and cross wind Error", self.MULE, play.DOWN, play.DISTANCE, play.YDLINE, play.playdesc)
                print(play.METAR.string())
                print(err)
                traceback.print_exc()
                print()
        return None
