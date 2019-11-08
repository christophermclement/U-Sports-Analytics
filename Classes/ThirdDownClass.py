# -*- coding: utf-8 -*-0
"""
Created on Sat Aug 25 19:34:30 2018

@author: Chris Clement
"""

import Globals
import Functions
import numpy
import itertools
import Classes.P1DClass as P1DClass
import Classes.EPClass as EPClass

import Classes.PuntClass as PuntClass
import Classes.FGClass as FGClass


class third_down():
    '''
    This contains the idea of a given down, distance, field position for third down decision-making.
    # TODO: We have a good object for going for it that encapsulates what we need from other objects and gives us consistent 
    # attribute naming, but we need a better way of handling safeties
    '''
    def __init__(self, distance, ydline):
        self.distance = numpy.int(distance)
        self.yardline = numpy.int(ydline)

        # Referring to the component objects
        self.punt= PuntClass.PUNT_ARRAY[self.yardline]
        self.field_goal = FGClass.FG_ARRAY[self.yardline]
        self.go_for_it = go_for_it_array[self.distance][self.yardline]
        
        choices_list = [self.punt, self.field_goal, self.go_for_it, Globals.score_values["SAFETY"]]
        self.choices_array = numpy.zeros((4, 4))
        return None

    def calculate(self):
        self.choices_list.sort(key=lambda x: x.EP[1])
        for perm, a, b in enumerate(itertools.permutations(self.choices_list, 2)):
            self.choices_array[a][b] = Functions.BootCompare(perm[0].EP_bootstrap, perm[1].EP_bootstrap)
        for c, choice in enumerate(self.choices_list):
            self.choices_array[c][c] = choice.EP[1]
        return None



class go_for_it():
    '''
    This is basically a client object of the third down, so we can better organize that object by having our three main choices have some consistent attributes, especially EP and EP_bootstrap
    # TODO: There's probably some fancy technique like "encapsulation" or some shit that we should be using?
    '''
    def __init__(self, distance, yardline):
        self.distance = distance
        self.yardline = yardline
        self.P1D = P1DClass.P1D_ARRAY[3][self.distance] if self.distance > self.yardline else P1DClass.P1D_GOAL_ARRAY[3][self.distance]
        self.EP_success = EPClass.EP_ARRAY[1][self.yardline - self.distance][self.yardline - self.distance] if self.distance == self.yardline else EPClass.EP_ARRAY[1][10][self.yardline - self.distance]
        #self.EP_fail = (EPClass.EP_ARRAY[1][110 - self.yardline][110 - self.yardline] if self.yardline > 100 else EPClass.EP_ARRAY[1][10][110 - self.yardline])

    def bootstrap(self):
        success_temp = Globals.score_bootstraps["TD"] if self.distance == self.yardline else numpy.repeat(self.EP_success.BOOTSTRAP, Globals.BOOTSTRAP_SIZE)
        fail_temp = numpy.transpose(numpy.repeat(self.EP_fail.BOOTSTRAP, Globals.BOOTSTRAP_SIZE))
        prob = numpy.divide(numpy.random.binomial(self.P1D.N, self.P1D.P[0], size=(Globals.BOOTSTRAP_SIZE, Globals.BOOTSTRAP_SIZE)), self.P1D.N)
        success_temp = numpy.multiply(success_temp, prob, out=success_temp)
        prob = numpy.subtract(1, prob, out=prob)
        fail_temp = numpy.multiply(fail_temp, prob, out=fail_temp)
        self.EP_bootstrap = numpy.subtract(success_temp, fail_temp, out=success_temp)
        self.EP_bootstrap = numpy.mean(self.EP_bootstrap, axis=1)
        self.EP_bootstrap = numpy.sort(self.EP_bootstrap)
        self.EP = numpy.array([self.EP_bootstrap[int(Globals.BOOTSTRAP_SIZE * Globals.CONFIDENCE - 1)],
                               numpy.mean(self.EP_bootstrap),
                               self.EP_bootstrap[int(Globals.BOOTSTRAP_SIZE * (1 - Globals.CONFIDENCE))]])


go_for_it_array = [[go_for_it(distance, yardline) for yardline in range(110)] for distance in range (110)]
third_down_array = [[third_down(distance, yardline) for yardline in range(110)] for distance in range (110)]


def go_for_it_boot():
    for distance in go_for_it_array:
        for yardline in distance:
            yardline.bootstrap()


def third_down_calculate():
    for distance in third_down_array:
        for yardline in third_down_array:
            yardline.calculate()
