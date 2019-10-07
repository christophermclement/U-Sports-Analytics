# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:48:35 2018

@author: Chris Clement
"""

import numpy
import pickle

class stadium():
    '''
    This is the class we use to hold the stadium objects we create with all
    their data and we can reference back to it because it's in the initial
    game statement.
    '''
    
    def __init__(self, name, city, province, home_teams, surface, capacity, GPS, orientation, elevation, airport, full_end_zones, TZCode, isDome):
        self.name = name  # Name of the stadium
        self.city = city  # City in which the stadium is located
        self.province = province  # Province in which the stadium is located
        self.home_teams = home_teams  # List of the teams that call this stadium home
        self.surface = surface  # Natural v turf and type of turf. TODO: Consider swapping this for a boolean "isNatural"
        self.capacity = capacity  # Seating capacity of the stadium, often approx due to renovations, etc.
        self.GPS = GPS  # Lat/Long of midfield, we could stand to cut down one decimal point but w/e
        self.orientation = orientation  # True bearing orientation of the long axis of the field, using the more northerly angle
        self.elevation = elevation  # Elevation above sea level in m
        self.airport = airport  # Gives the nearest airport code to use the proper METAR data
        self.full_end_zones = full_end_zones  # True if the end zones are full sized, false if they are cut off (not full length, corners cut bc of track, etc.
        self.TZCode = TZCode  # Gives the appropriate code to handle time zones
        self.isDome = isDome  # True if it's a domed stadium
        return None

stadium_list =\
[['Oland Stadium', 'Antigonish', 'NS', ['SFX'], 'FieldTurf', 4000, [45.616639, -61.994972], 359, 195, 'CYYG', False, "America/Halifax", False],
 ['Huskies Stadium', 'Halifax', 'NS', ['SMU'], 'FieldTurf', 5000, [44.631111, -63.579556], 343, 23, 'CYHZ', False, "America/Halifax", False],
 ['Raymond Field', 'Wolfville', 'NS', ['ACA'], 'FieldTurf', 3000, [45.092000, -64.367444], 346, 8, 'CYHZ', False, "America/Halifax", False],
 ['MacAulay Field', 'Sackville', 'NB', ['MTA'], 'Natural Grass', 2500, [45.897750, -64.373389], 281, 18, 'CYQM', False, "America/Halifax", False],
 ['Coulter Field', 'Lennoxville', 'QC', ['BIS'], 'FieldTurf', 2200, [45.365222, -71.841194], 0, 148, 'CYSC', True, "America/Toronto", False],
 ["Stade de l'Universite", 'Sherbrooke', 'QC', ['SHE'], '', 3359, [45.374583, -71.930778], 271, 252, 'CYSC', False, "America/Toronto", False],
 ['CEPSUM', 'Montreal', 'QC', ['MON'], '', 5100, [45.509083, -73.611472], 32, 133, 'CYUL', True, "America/Toronto", False],
 ['Concordia Stadium', 'Montreal', 'QC', ['CON'], 'FieldTurf', 4000, [45.457972, -73.63747], 39, 54, 'CYUL', True, "America/Toronto", False],
 ['Percival Molson Stadium', 'Montreal', 'QC', ['MCG, ALS'], 'FieldTurf', 23420, [45.510111, -73.580889], 10, 66, 'CYUL', False, "America/Toronto", False],
 ['Stade TELUS', 'Quebec', 'QC', ['LAV'], 'FieldTurf', 12817, [46.783722, -71.279611], 321, 81, 'CYQB', False, "America/Toronto", False],
 ['MNP Park', 'Ottawa', 'ON', ['CAR'], 'FieldTurf', '3500', [45.388583, -75.694194], 337, 64, 'CYOW', True, "America/Toronto", False],
 ['Gee-Gees Field', 'Ottawa', 'ON', ['OTT'], 'FieldTurf', 3000, [45.416056, -75.665194], 88, 61, 'CYOW', False, "America/Toronto", False],
 ['Old Richardson Memorial Stadium', 'Kingston', 'ON', ['QUE'], 'Natural Grass', 8000, [44.227639, -76.516333], 356, 93, 'CYGK', False, "America/Toronto", False],
 ['New Richardson Memorial Stadium', 'Kingston', 'ON', ['QUE'], 'FieldTurf', 8000, [44.227639, -76.516333], 356, 93, 'CYGK', True, "America/Toronto", False],
 ['Varsity Stadium', 'Toronto', 'ON', ['TOR'], 'Polytan Ligaturf', 5000, [43.667056, -79.397278], 344, 111, 'CYTZ', False, "America/Toronto", False],
 ['York Lions Stadium', 'Toronto', 'ON', ['YRK'], 'FieldTurf', 3700, [43.776417, -79.511972], 345, 199, 'CYYZ', True, "America/Toronto", False],
 ['Ron Joyce Stadium', 'Hamilton', 'ON', ['MAC'], 'FieldTurf', 6000, [43.266056, -79.917028], 358, 90, 'CYHM', True, "America/Toronto", False],
 ['Guelph Alumni Stadium', 'Guelph', 'ON', ['GUE'], 'FieldTurf Revolution', 8500, [43.535056, -80.226583], 317, 336, 'CYYZ', False, "America/Toronto", False],
 ['Warrior Field', 'Waterloo', 'ON', ['WAT'], 'FieldTruf Duraspin PRO', 5400, [43.474250, -80.549694], 64, 342, 'CYKF', True, "America/Toronto", False],
 ['University Stadium', 'Waterloo', 'ON', ['WLU'], 'FieldTurf', 6000, [43.470194, -80.53013], 63, 332, 'CYKF', False, "America/Toronto", False],
 ['TD Waterhouse Stadium', 'London', 'ON', ['WES'], 'FieldTurf', 8000, [42.999889, -81.273833], 8, 249, 'CYXU', False, "America/Toronto", False],
 ['University of Windsor Stadium', 'Windsor', 'ON', ['WIN'], 'FieldTurf', 2000, [42.298222, -83.063000], 8, 180, 'CYQG', False, "America/Toronto", False],
 ['Investors Group Field', 'Winnipeg', 'MB', ['MAN, BBO'], 'FieldTurf', 33500, [49.807833, -97.143000], 0, 227, 'CYWG', True, "America/Winnipeg", False],
 ['University of Manitoba Stadium', 'Winnipeg', 'MB', ['MAN'], 'Natural Grass', 5000, [49.806750, -97.146222], 332, 227, 'CYWG', False, "America/Winnipeg", False],
 ['Griffiths Stadium', 'Saskatoon', 'SK', ['SKH'], 'FieldTurf', 6171, [52.127083, -106.629833], 355, 502, 'CYXE', False, "America/Regina", False],
 ['Mosaic Stadium', 'Regina', 'SK', ['REG, RRI'], 'FieldTurf', 33427, [50.450528, -104.633083], 355, 576, 'CYQR', True, "America/Regina", False],
 ['Mosaic Stadium at Taylor Field', 'Regina', 'SK', ['REG, RRI'], 'FieldTurf', 33350, [50.452639, -104.624222], 315, 576, 'CYQR', True, "America/Regina", False],
 ['McMahon Stadium', 'Calgary', 'AB', ['CGY, STA'], 'FieldTurf', 33650, [51.070389, -114.121472], 335, 1099, 'CYYC', True, "America/Edmonton", False],
 ['Foote Field', 'Edmonton', 'AB', ['ALB'], 'PureGrass', 3500, [53.503528, -113.530472], 0, 669, 'CYEG', True, "America/Edmonton", False],
 ['Thunderbird Stadium', 'Vancouver', 'BC', ['UBC'], 'PolyTan Turf', 3411, [49.254417, -123.245556], 331, 79, 'CYVR', True, "America/Vancouver", False],
 ['Swangard Stadium', 'Burnaby', 'BC', ['SFU'], 'Natural Grass', 5288, [49.278639, -122.922278], 103, 323, 'CYVR', True, "America/Vancouver", False],
 ['TD Place', 'Ottawa', 'ON', ['RED'], 'FieldTurf', 24000, [45.398194, -75.683472], 60, 63, 'CYOW', True, "America/Toronto", False],
 ['Setters Place', 'Red Deer', 'AB', [], 'Natural Grass', 500, [52.268028, -113.833472], 0, 855, 'CYQF', False, "America/Edmonton", False],
 ['Westhills Stadium', 'Victoria', 'BC', ['VIR'], 'FieldTurf', 1718, [48.443083, -123.523611], 274, 69, 'CYYJ', True, "America/Vancouver", False],
 ['Rogers Centre', 'Toronto', 'ON', ['ARG'], 'AstroTurf', 53506, [43.641500, -79.389139], 345, 79, 'CYYZ', True, "America/Toronto", False],
 ['Commonwealth Stadium', 'Edmonton', 'AB', ['ESK'], 'Shaw Sports Turf', 56302,  [53.559611, -113.476167], 0, 657, 'CYEG', False, "America/Edmonton", False],
 ['BMO Stadium', 'Toronto', 'ON', ['ARG'], 'Natural Grass', 30991, [43.633222, -79.418583], 344, 83, 'CYTZ', True, "America/Toronto", False],
 ['Olympic Stadium', 'Montreal', 'QC', ['ALS'], 'FieldTurf', 56040, [45.557917, -73.551500], 350, 27, 'CYUL', True, "America/Toronto", False],
 ['BC Place', 'Vancouver', 'BC', ['LNS'], 'FieldTurf', 54320, [49.276639, -123.111861], 54, 8, 'CYVR', True, "America/Vancouver", False],
 ['Tim Hortons Field', 'Hamiton', 'ON', ['TIC'], 'FieldTurf', 24000, [43.252111, -79.830083], 305, 85, 'CYHM', True, "America/Toronto", False],
 ['Moncton Stadium', 'Moncton', 'NB', [], 'FieldTurf', 10000, [46.108528, -64.78350], 29, 18, 'CYQM', False, "America/Halifax", False],
 ['Ivor Wynne Stadium', 'Hamilton', 'ON', ['TIC'], 'AstroPlay', 29600, [43.252111, -79.830083], 107, 85, 'CYHM', True, "America/Toronto", False],
 ['CanadInns Stadium', 'Winnipeg', 'MB', ['BBO'], 'AstroPlay', 29533, [49.890120, -97.197260], 4, 231, 'CYWG', True, "America/Winnipeg", False]]

stadia = {x[0] : stadium(*x) for x in stadium_list}

class scoring_play():
    '''
    Let's see if we make a class for the four different scoring plays, that way we have the EP and EP_bootstrap
    '''
    def __init__(self, name, base_value):
        self.name = name  # This is like "TD" or "ROUGE" or whatever
        self.EP = numpy.empty(3, dtype='Float64')
        self.EP[1] = base_value
        self.EP_bootstrap = None

high_accuracy = True  # This is a quick boolean to drastically reduce the degree of precision of our modeling for testing purposes
if high_accuracy:
    BOOTSTRAP_SIZE = 1000  # Number of bootstrap iterations to use
    forest_trees = 1000
    neural_network = (100, 100, 100)
    KFolds = 10
else:
    BOOTSTRAP_SIZE = 200  # Number of bootstrap iterations to use
    forest_trees = 100
    neural_network = (20, 20, 20)
    KFolds = 3


CONFIDENCE = 0.025  # The one-sided confidence interval size for all statistical tests
THRESHOLD = 100  # The minimum N to include in the graphs
DISTANCE_LIMIT = 51  # This is the max size of P(1D), etc. we're accepting. Usually 26 (meaning 25) but we'll try with longer ones.
alpha_scores = [["FG", False], ["ROUGE", False], ["SAFETY", False], ["TD", False], ["HALF", False], ["FG", True], ["ROUGE", True], ["SAFETY", True], ["TD", True]]

score_values = {"FG": scoring_play("FG", 3),
                "ROUGE": scoring_play("ROUGE", 2),
                "SAFETY": scoring_play("SAFETY", -2),
                "TD": scoring_play("TD", 7),
                "HALF": scoring_play("HALF", 0)}

'''
score_bootstraps = {"FG": numpy.empty(BOOTSTRAP_SIZE, dtype=numpy.dtype('Float64')),
                    "ROUGE": numpy.empty(BOOTSTRAP_SIZE, dtype=numpy.dtype('Float64')),
                    "SAFETY": numpy.empty(BOOTSTRAP_SIZE, dtype=numpy.dtype('Float64')),
                    "TD": numpy.empty(BOOTSTRAP_SIZE, dtype=numpy.dtype('Float64')),
                    "HALF": numpy.empty(BOOTSTRAP_SIZE, dtype=numpy.dtype('Float64'))}
'''

gamelist = []  # The gamelist holds all the games

# Holds a default value to avoid errors when comparing to None
DummyArray = numpy.full(BOOTSTRAP_SIZE, -100, dtype='int32')

passerList = []
receiverList = []
tacklerList= []

CISTeams = ["SFX", "SMU", "ACA", "MTA", "BIS",
            "SHE", "LAV", "MON", "CON", "MCG",
            "OTT", "CAR", "QUE", "TOR", "YRK", "MAC", "GUE", "WAT", "WLU", "WES", "WIN",
            "MAN", "REG", "SKH", "ALB", "CGY", "UBC", "SFU"]
CISConferences = ["AUS", "RSEQ", "OUA", "CWUAA", "NONCON"]