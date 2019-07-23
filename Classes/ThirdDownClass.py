<<<<<<< HEAD
<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:34:30 2018

@author: Chris Clement
"""

import Globals
import Functions
import numpy
import Classes.P1DClass as P1DClass
import Classes.EPClass as EPClass

import Classes.PuntClass as PuntClass
import Classes.FGClass as FGClass

THIRD_DOWN_ARRAY = []


class THIRD_DOWN():
    def __init__(self, distance, ydline):
        self.DISTANCE = distance
        self.YDLINE = ydline

        self.PUNT = None
        self.PUNT_BOOTSTRAP = Globals.DummyArray
        self.PUNT_HIGH = self.PUNT_LOW = None

        self.FIELDGOAL = self.FIELDGOAL_LOW = self.FIELDGOAL_HIGH = None
        self.FIELDGOAL_BOOTSTRAP = Globals.DummyArray

        self.SAFETY = self.SAFETY_LOW = self.SAFETY_HIGH = None
        self.SAFETY_BOOTSTRAP = None

        self.GOFORIT = self.P1D = self.CONVERT = self.FAIL = None
        self.GOFORIT_BOOTSTRAP = self.CONVERT_BOOTSTRAP = self.FAIL_BOOTSTRAP = None

        self.GOvsPUNT = self.PUNTvsGO = 0
        self.GOvsKICK = self.KICKvsGO = 0
        self.GOvsSAFETY = self.SAFETYvsGO = 0
        self.PUNTvsKICK = self.KICKvsPUNT = 0
        self.PUNTvsSAFETY = self.SAFETYvsPUNT = 0
        self.KICKvsSAFETY = self.SAFETYvsKICK = 0

        self.GOvsALL = self.PUNTvsALL = self.KICKvsALL = self.SAFETYvsALL = 0

    def calculate(self):

        self.SAFETY = Globals.SCOREvals[2][1]
        self.SAFETY_LOW = Globals.SCOREvals[2][0]
        self.SAFETY_HIGH = Globals.SCOREvals[2][2]
        self.SAFETY_BOOTSTRAP = Globals.SAFETYval_BOOTSTRAP

        if self.YDLINE > 0 and self.YDLINE >= self.DISTANCE > 0 and self.YDLINE - self.DISTANCE < 100:
            if self.DISTANCE == self.YDLINE:  # For & Goal situations
                self.CONVERT = Globals.SCOREvals[3][1]  # Converting means scoring a TD
                self.CONVERT_BOOTSTRAP = Globals.TDval_BOOTSTRAP
            # When converting puts you &Goal
            elif self.YDLINE - self.DISTANCE < 10:
                self.CONVERT = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].BOOTSTRAP
            else:  # Normal situations
                self.CONVERT =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].BOOTSTRAP

            # When backed up so much that failing means 1st & Goal - impossible
            if self.YDLINE > 100:
                self.FAIL = EPClass.EP_ARRAY[1][110-self.YDLINE][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][110 - self.YDLINE][110 - self.YDLINE].BOOTSTRAP
            else:
                self.FAIL = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].BOOTSTRAP
            
            self.P1D = P1DClass.P1D_ARRAY[3][self.DISTANCE].P
            if self.P1D is None:
                self.P1D = ((P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N)
                             / (P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N))

            self.GOFORIT = self.CONVERT * self.P1D - (1 - self.P1D) * self.FAIL

        self.PUNT = PuntClass.PUNT_ARRAY[self.YDLINE].EP
        self.PUNT_BOOTSTRAP = PuntClass.PUNT_ARRAY[self.YDLINE].BOOTSTRAP
        self.PUNT_HIGH = PuntClass.PUNT_ARRAY[self.YDLINE].EP_HIGH
        self.PUNT_LOW = PuntClass.PUNT_ARRAY[self.YDLINE].EP_LOW

        self.FIELDGOAL = FGClass.FG_ARRAY[self.YDLINE].EP
        self.FIELDGOAL_LOW = FGClass.FG_ARRAY[self.YDLINE].EP_LOW
        self.FIELDGOAL_HIGH = FGClass.FG_ARRAY[self.YDLINE].EP_HIGH
        self.FIELDGOAL_BOOTSTRAP = FGClass.FG_ARRAY[self.YDLINE].BOOTSTRAP

    def boot(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:
            poned = numpy.random.binomial(P1DClass.P1D_ARRAY[3][self.DISTANCE].N, self.P1D, Globals.BOOTSTRAP_SIZE ** 2)
            poned = poned / P1DClass.P1D_ARRAY[3][self.DISTANCE].N
            self.CONVERT_BOOTSTRAP =\
                numpy.repeat(self.CONVERT_BOOTSTRAP, Globals.BOOTSTRAP_SIZE)
            self.FAIL_BOOTSTRAP =\
                numpy.repeat(self.FAIL_BOOTSTRAP, Globals.BOOTSTRAP_SIZE, axis=0)
            self.GOFORIT_BOOTSTRAP = (self.CONVERT_BOOTSTRAP*poned + ((poned + (-1)) * self.FAIL_BOOTSTRAP))
            self.GOFORIT_BOOTSTRAP = numpy.sort(numpy.mean(self.GOFORIT_BOOTSTRAP.reshape(-1, Globals.BOOTSTRAP_SIZE), 1, dtype='f4'))

    def compare(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:

            self.GOvsSAFETY = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                    self.SAFETY_BOOTSTRAP)
            self.SAFETYvsGO = 1 - self.GOvsSAFETY
            self.PUNTvsKICK = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                    self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsPUNT = 1 - self.PUNTvsKICK
            self.PUNTvsSAFETY = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsPUNT = 1 - self.PUNTvsSAFETY
            self.GOvsPUNT = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.PUNT_BOOTSTRAP)
            self.PUNTvsGO = 1 - self.GOvsPUNT
            self.GOvsKICK = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsGO = 1 - self.GOvsKICK
            self.KICKvsSAFETY = Functions.BootCompare(self.FIELDGOAL_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsKICK = 1 - self.KICKvsSAFETY

            self.GOvsALL = (self.GOvsPUNT + self.GOvsKICK +
                            self.GOvsSAFETY) / 3
            self.SAFETYvsALL = (self.SAFETYvsGO + self.SAFETYvsPUNT +
                                self.SAFETYvsKICK) / 3
            self.PUNTvsALL = (self.PUNTvsGO + self.PUNTvsKICK +
                              self.PUNTvsSAFETY) / 3
            self.KICKvsALL = (self.KICKvsGO + self.KICKvsPUNT +
                              self.KICKvsSAFETY) / 3


def Array_Declaration():
    global THIRD_DOWN_ARRAY
    THIRD_DOWN_ARRAY = []
    for distance in range(Globals.DISTANCE_LIMIT + 1):
        temp = []
        for yardline in range(110):
            temp.append(THIRD_DOWN(distance, yardline))
        THIRD_DOWN_ARRAY.append(temp)


def ThirdDown_calculate():
    for DISTANCE in THIRD_DOWN_ARRAY:
        for YDLINE in DISTANCE:
            YDLINE.calculate()


def ThirdDown_boot():
    for distance in THIRD_DOWN_ARRAY:
        for ydline in distance:
            ydline.boot()
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:34:30 2018

@author: Chris Clement
"""

import Globals
import Functions
import numpy
import Classes.P1DClass as P1DClass
import Classes.EPClass as EPClass

import Classes.PuntClass as PuntClass
import Classes.FGClass as FGClass

THIRD_DOWN_ARRAY = []


class THIRD_DOWN():
    def __init__(self, distance, ydline):
        self.DISTANCE = distance
        self.YDLINE = ydline

        self.PUNT = None
        self.PUNT_BOOTSTRAP = Globals.DummyArray
        self.PUNT_HIGH = self.PUNT_LOW = None

        self.FIELDGOAL = self.FIELDGOAL_LOW = self.FIELDGOAL_HIGH = None
        self.FIELDGOAL_BOOTSTRAP = Globals.DummyArray

        self.SAFETY = self.SAFETY_LOW = self.SAFETY_HIGH = None
        self.SAFETY_BOOTSTRAP = None

        self.GOFORIT = self.P1D = self.CONVERT = self.FAIL = None
        self.GOFORIT_BOOTSTRAP = self.CONVERT_BOOTSTRAP = self.FAIL_BOOTSTRAP = None

        self.GOvsPUNT = self.PUNTvsGO = 0
        self.GOvsKICK = self.KICKvsGO = 0
        self.GOvsSAFETY = self.SAFETYvsGO = 0
        self.PUNTvsKICK = self.KICKvsPUNT = 0
        self.PUNTvsSAFETY = self.SAFETYvsPUNT = 0
        self.KICKvsSAFETY = self.SAFETYvsKICK = 0

        self.GOvsALL = self.PUNTvsALL = self.KICKvsALL = self.SAFETYvsALL = 0

    def calculate(self):

        self.SAFETY = Globals.SCOREvals[2][1]
        self.SAFETY_LOW = Globals.SCOREvals[2][0]
        self.SAFETY_HIGH = Globals.SCOREvals[2][2]
        self.SAFETY_BOOTSTRAP = Globals.SAFETYval_BOOTSTRAP

        if self.YDLINE > 0 and self.YDLINE >= self.DISTANCE > 0 and self.YDLINE - self.DISTANCE < 100:
            if self.DISTANCE == self.YDLINE:  # For & Goal situations
                self.CONVERT = Globals.SCOREvals[3][1]  # Converting means scoring a TD
                self.CONVERT_BOOTSTRAP = Globals.TDval_BOOTSTRAP
            # When converting puts you &Goal
            elif self.YDLINE - self.DISTANCE < 10:
                self.CONVERT = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].BOOTSTRAP
            else:  # Normal situations
                self.CONVERT =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].BOOTSTRAP

            # When backed up so much that failing means 1st & Goal - impossible
            if self.YDLINE > 100:
                self.FAIL = EPClass.EP_ARRAY[1][110-self.YDLINE][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][110 - self.YDLINE][110 - self.YDLINE].BOOTSTRAP
            else:
                self.FAIL = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].BOOTSTRAP
            
            self.P1D = P1DClass.P1D_ARRAY[3][self.DISTANCE].P
            if self.P1D is None:
                self.P1D = ((P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N)
                             / (P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N))

            self.GOFORIT = self.CONVERT * self.P1D - (1 - self.P1D) * self.FAIL

        self.PUNT = PuntClass.PUNT_ARRAY[self.YDLINE].EP
        self.PUNT_BOOTSTRAP = PuntClass.PUNT_ARRAY[self.YDLINE].BOOTSTRAP
        self.PUNT_HIGH = PuntClass.PUNT_ARRAY[self.YDLINE].EP_HIGH
        self.PUNT_LOW = PuntClass.PUNT_ARRAY[self.YDLINE].EP_LOW

        self.FIELDGOAL = FGClass.FG_ARRAY[self.YDLINE].EP
        self.FIELDGOAL_LOW = FGClass.FG_ARRAY[self.YDLINE].EP_LOW
        self.FIELDGOAL_HIGH = FGClass.FG_ARRAY[self.YDLINE].EP_HIGH
        self.FIELDGOAL_BOOTSTRAP = FGClass.FG_ARRAY[self.YDLINE].BOOTSTRAP

    def boot(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:
            poned = numpy.random.binomial(P1DClass.P1D_ARRAY[3][self.DISTANCE].N, self.P1D, Globals.BOOTSTRAP_SIZE ** 2)
            poned = poned / P1DClass.P1D_ARRAY[3][self.DISTANCE].N
            self.CONVERT_BOOTSTRAP =\
                numpy.repeat(self.CONVERT_BOOTSTRAP, Globals.BOOTSTRAP_SIZE)
            self.FAIL_BOOTSTRAP =\
                numpy.repeat(self.FAIL_BOOTSTRAP, Globals.BOOTSTRAP_SIZE, axis=0)
            self.GOFORIT_BOOTSTRAP = (self.CONVERT_BOOTSTRAP*poned + ((poned + (-1)) * self.FAIL_BOOTSTRAP))
            self.GOFORIT_BOOTSTRAP = numpy.sort(numpy.mean(self.GOFORIT_BOOTSTRAP.reshape(-1, Globals.BOOTSTRAP_SIZE), 1, dtype='f4'))

    def compare(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:

            self.GOvsSAFETY = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                    self.SAFETY_BOOTSTRAP)
            self.SAFETYvsGO = 1 - self.GOvsSAFETY
            self.PUNTvsKICK = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                    self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsPUNT = 1 - self.PUNTvsKICK
            self.PUNTvsSAFETY = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsPUNT = 1 - self.PUNTvsSAFETY
            self.GOvsPUNT = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.PUNT_BOOTSTRAP)
            self.PUNTvsGO = 1 - self.GOvsPUNT
            self.GOvsKICK = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsGO = 1 - self.GOvsKICK
            self.KICKvsSAFETY = Functions.BootCompare(self.FIELDGOAL_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsKICK = 1 - self.KICKvsSAFETY

            self.GOvsALL = (self.GOvsPUNT + self.GOvsKICK +
                            self.GOvsSAFETY) / 3
            self.SAFETYvsALL = (self.SAFETYvsGO + self.SAFETYvsPUNT +
                                self.SAFETYvsKICK) / 3
            self.PUNTvsALL = (self.PUNTvsGO + self.PUNTvsKICK +
                              self.PUNTvsSAFETY) / 3
            self.KICKvsALL = (self.KICKvsGO + self.KICKvsPUNT +
                              self.KICKvsSAFETY) / 3


def Array_Declaration():
    global THIRD_DOWN_ARRAY
    THIRD_DOWN_ARRAY = []
    for distance in range(Globals.DISTANCE_LIMIT + 1):
        temp = []
        for yardline in range(110):
            temp.append(THIRD_DOWN(distance, yardline))
        THIRD_DOWN_ARRAY.append(temp)


def ThirdDown_calculate():
    for DISTANCE in THIRD_DOWN_ARRAY:
        for YDLINE in DISTANCE:
            YDLINE.calculate()


def ThirdDown_boot():
    for distance in THIRD_DOWN_ARRAY:
        for ydline in distance:
            ydline.boot()
>>>>>>> parent of 7093df1... Merge branch 'master' of https://github.com/christophermclement/U-Sports-Analytics
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:34:30 2018

@author: Chris Clement
"""

import Globals
import Functions
import numpy
import Classes.P1DClass as P1DClass
import Classes.EPClass as EPClass

import Classes.PuntClass as PuntClass
import Classes.FGClass as FGClass

THIRD_DOWN_ARRAY = []


class THIRD_DOWN():
    def __init__(self, distance, ydline):
        self.DISTANCE = distance
        self.YDLINE = ydline

        self.PUNT = None
        self.PUNT_BOOTSTRAP = Globals.DummyArray
        self.PUNT_HIGH = self.PUNT_LOW = None

        self.FIELDGOAL = self.FIELDGOAL_LOW = self.FIELDGOAL_HIGH = None
        self.FIELDGOAL_BOOTSTRAP = Globals.DummyArray

        self.SAFETY = self.SAFETY_LOW = self.SAFETY_HIGH = None
        self.SAFETY_BOOTSTRAP = None

        self.GOFORIT = self.P1D = self.CONVERT = self.FAIL = None
        self.GOFORIT_BOOTSTRAP = self.CONVERT_BOOTSTRAP = self.FAIL_BOOTSTRAP = None

        self.GOvsPUNT = self.PUNTvsGO = 0
        self.GOvsKICK = self.KICKvsGO = 0
        self.GOvsSAFETY = self.SAFETYvsGO = 0
        self.PUNTvsKICK = self.KICKvsPUNT = 0
        self.PUNTvsSAFETY = self.SAFETYvsPUNT = 0
        self.KICKvsSAFETY = self.SAFETYvsKICK = 0

        self.GOvsALL = self.PUNTvsALL = self.KICKvsALL = self.SAFETYvsALL = 0

    def calculate(self):

        self.SAFETY = Globals.SCOREvals[2][1]
        self.SAFETY_LOW = Globals.SCOREvals[2][0]
        self.SAFETY_HIGH = Globals.SCOREvals[2][2]
        self.SAFETY_BOOTSTRAP = Globals.SAFETYval_BOOTSTRAP

        if self.YDLINE > 0 and self.YDLINE >= self.DISTANCE > 0 and self.YDLINE - self.DISTANCE < 100:
            if self.DISTANCE == self.YDLINE:  # For & Goal situations
                self.CONVERT = Globals.SCOREvals[3][1]  # Converting means scoring a TD
                self.CONVERT_BOOTSTRAP = Globals.TDval_BOOTSTRAP
            # When converting puts you &Goal
            elif self.YDLINE - self.DISTANCE < 10:
                self.CONVERT = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP = EPClass.EP_ARRAY[1][self.YDLINE - self.DISTANCE][self.YDLINE - self.DISTANCE].BOOTSTRAP
            else:  # Normal situations
                self.CONVERT =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].EP
                self.CONVERT_BOOTSTRAP =\
                    EPClass.EP_ARRAY[1][10][self.YDLINE - self.DISTANCE].BOOTSTRAP

            # When backed up so much that failing means 1st & Goal - impossible
            if self.YDLINE > 100:
                self.FAIL = EPClass.EP_ARRAY[1][110-self.YDLINE][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][110 - self.YDLINE][110 - self.YDLINE].BOOTSTRAP
            else:
                self.FAIL = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].EP
                self.FAIL_BOOTSTRAP = EPClass.EP_ARRAY[1][10][110 - self.YDLINE].BOOTSTRAP
            
            self.P1D = P1DClass.P1D_ARRAY[3][self.DISTANCE].P
            if self.P1D is None:
                self.P1D = ((P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].P
                             * P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N)
                             / (P1DClass.P1D_ARRAY[3][self.DISTANCE - 1].N
                             + P1DClass.P1D_ARRAY[3][self.DISTANCE + 1].N))

            self.GOFORIT = self.CONVERT * self.P1D - (1 - self.P1D) * self.FAIL

        self.PUNT = PuntClass.PUNT_ARRAY[self.YDLINE].EP
        self.PUNT_BOOTSTRAP = PuntClass.PUNT_ARRAY[self.YDLINE].BOOTSTRAP
        self.PUNT_HIGH = PuntClass.PUNT_ARRAY[self.YDLINE].EP_HIGH
        self.PUNT_LOW = PuntClass.PUNT_ARRAY[self.YDLINE].EP_LOW

        self.FIELDGOAL = FGClass.FG_ARRAY[self.YDLINE].EP
        self.FIELDGOAL_LOW = FGClass.FG_ARRAY[self.YDLINE].EP_LOW
        self.FIELDGOAL_HIGH = FGClass.FG_ARRAY[self.YDLINE].EP_HIGH
        self.FIELDGOAL_BOOTSTRAP = FGClass.FG_ARRAY[self.YDLINE].BOOTSTRAP

    def boot(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:
            poned = numpy.random.binomial(P1DClass.P1D_ARRAY[3][self.DISTANCE].N, self.P1D, Globals.BOOTSTRAP_SIZE ** 2)
            poned = poned / P1DClass.P1D_ARRAY[3][self.DISTANCE].N
            self.CONVERT_BOOTSTRAP =\
                numpy.repeat(self.CONVERT_BOOTSTRAP, Globals.BOOTSTRAP_SIZE)
            self.FAIL_BOOTSTRAP =\
                numpy.repeat(self.FAIL_BOOTSTRAP, Globals.BOOTSTRAP_SIZE, axis=0)
            self.GOFORIT_BOOTSTRAP = (self.CONVERT_BOOTSTRAP*poned + ((poned + (-1)) * self.FAIL_BOOTSTRAP))
            self.GOFORIT_BOOTSTRAP = numpy.sort(numpy.mean(self.GOFORIT_BOOTSTRAP.reshape(-1, Globals.BOOTSTRAP_SIZE), 1, dtype='f4'))

    def compare(self):
        if self.YDLINE > 0 and 0 < self.DISTANCE <= self.YDLINE and self.YDLINE - self.DISTANCE < 100:

            self.GOvsSAFETY = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                    self.SAFETY_BOOTSTRAP)
            self.SAFETYvsGO = 1 - self.GOvsSAFETY
            self.PUNTvsKICK = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                    self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsPUNT = 1 - self.PUNTvsKICK
            self.PUNTvsSAFETY = Functions.BootCompare(self.PUNT_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsPUNT = 1 - self.PUNTvsSAFETY
            self.GOvsPUNT = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.PUNT_BOOTSTRAP)
            self.PUNTvsGO = 1 - self.GOvsPUNT
            self.GOvsKICK = Functions.BootCompare(self.GOFORIT_BOOTSTRAP,
                                                  self.FIELDGOAL_BOOTSTRAP)
            self.KICKvsGO = 1 - self.GOvsKICK
            self.KICKvsSAFETY = Functions.BootCompare(self.FIELDGOAL_BOOTSTRAP,
                                                      self.SAFETY_BOOTSTRAP)
            self.SAFETYvsKICK = 1 - self.KICKvsSAFETY

            self.GOvsALL = (self.GOvsPUNT + self.GOvsKICK +
                            self.GOvsSAFETY) / 3
            self.SAFETYvsALL = (self.SAFETYvsGO + self.SAFETYvsPUNT +
                                self.SAFETYvsKICK) / 3
            self.PUNTvsALL = (self.PUNTvsGO + self.PUNTvsKICK +
                              self.PUNTvsSAFETY) / 3
            self.KICKvsALL = (self.KICKvsGO + self.KICKvsPUNT +
                              self.KICKvsSAFETY) / 3


def Array_Declaration():
    global THIRD_DOWN_ARRAY
    THIRD_DOWN_ARRAY = []
    for distance in range(Globals.DISTANCE_LIMIT + 1):
        temp = []
        for yardline in range(110):
            temp.append(THIRD_DOWN(distance, yardline))
        THIRD_DOWN_ARRAY.append(temp)


def ThirdDown_calculate():
    for DISTANCE in THIRD_DOWN_ARRAY:
        for YDLINE in DISTANCE:
            YDLINE.calculate()


def ThirdDown_boot():
    for distance in THIRD_DOWN_ARRAY:
        for ydline in distance:
            ydline.boot()
>>>>>>> parent of 7093df1... Merge branch 'master' of https://github.com/christophermclement/U-Sports-Analytics
