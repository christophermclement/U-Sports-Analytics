# -*- coding: utf-8 -*-
"""
Created on Tue Oct 08 10:44:25 2019

@author: Chris Clement
"""


import csv


def nameswaps():
    '''
    function to clean up the list of names in the database for the passerList, rusherList, etc.
    '''

    changes = {}  # dictionary of keys to find and values to replace

    for mule in range(1, 4):
        with open("Data/CIS Mule 0" + mule.str() + ".csv", 'rb') as f:
            reader = csv.reader(f) # pass the file to our csv reader
            for row in reader:     # iterate over the rows in the file
                new_row = row      # at first, just copy the row
                for key, value in changes.items(): # iterate over 'changes' dictionary
                    new_row = [ x.replace(key, value) for x in new_row ] # make the substitutions
                new_rows.append(new_row) # add the modified rows

        with open("Data/New CIS Mule 0" + mule.str() + ".csv", 'wb') as f:
            # Overwrite the old file with the modified rows
            writer = csv.writer(f)
            writer.writerows(new_rows)
        
