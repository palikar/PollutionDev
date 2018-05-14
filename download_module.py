#!/usr/bin/python



import utils as ut
import sys
import json
import numpy as np
from bs4 import BeautifulSoup
import urllib
import re


base_url =  None
start_date = None
end_date = None


def main():
    print("Starting downlaod module from command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    base_url = config_data["download_module"]["base_url"]#
    print(base_url)

    html = urllib.request.urlopen("https://archive.luftdaten.info/2017-01-01/").read()
    soup = BeautifulSoup(html, "lxml")

    
    for link in soup.findAll("a"):
        if re.match(".*.csv", link.get("href")):
            print( link.get("href"))
            

        
    
    dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-12-31'))
    # print(dates)
    
    
if __name__ == '__main__':
    main()
