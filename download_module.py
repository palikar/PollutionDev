#!/usr/bin/python



import utils as ut
import sys
import json
import numpy as np


base_url =  None



def main():
    print("Starting downlaod module from command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    base_url = config_data["download_module"]["base_url"]#
    print(base_url)

    dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-12-31'))
    print(dates)
    
    
if __name__ == '__main__':
    main()
