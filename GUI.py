# -*- coding: utf-8 -*-0
"""
Created on Mon Sep 23 22:00:30 2019

@author: Chris Clement
"""

import tkinter
from Classes import ThirdDownClass
import WP
import Globals

main_menu = None
third_down_menu = None

class game_calculator():
    '''
    Sets up the tkinter GUI for the in-game calculator
    '''
    def __init__(self, master):
        self.master = master
        master.title("Game calculator")
        
        self.away_score = tkinter.Entry(master)
        self.away_score.grid(1, 1)
        self.home_score = tkinter.Entry(master)
        self.home_score.grid(1, 3)

        self.time = tkinter.Entry(master)
        self.time.grid(1, 2)
        #for qtr in range(5):  # TODO: Make quarter into a radiobutton or maybe a pulldown


        self.quarter = tkinter.Entry(master)
        self.quarter.grid(2, 2)

        self.away_timeouts = tkinter.Entry(master)
        self.away_timeouts.pack(2, 1)
        self.home_timeouts = tkinter.Entry(master)
        self.home_timeouts.pack(2, 3)

        self.down = tkinter.Entry(master)
        self.down.pack(3, 1)
        self.distance = tkinter.Entry(master)
        self.distance.pack(3, 2)
        self.yardline = tkinter.Entry(master)
        self.yardline.pack(3, 3)

        self.calculate_button = tkinter.Button(master, command=self.calculate)
        self.output = tkinter.Label(master)
    
    def calculate(self):
        '''
        # TODO: Include a fuckton of error-checking
        '''
        time = 3600 - 900 * self.quarter + self.time.get()[:-2] * 90 + self.time.get()[-2]
        data = [self.down.get(),
                self.distance.get(),
                self.yardline.get(),
                time,
                ]




def GUI_go():
    root = tkinter.Tk()
    game_gui = game_calculator(root)
    game_gui.mainloop()
