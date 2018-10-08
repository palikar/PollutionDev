#!/usr/bin/python



import utils as ut
import sys
import json
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import os
import wget


base_url =  None
download_dir = None
start_date = None 
end_date = None
sensor_type = None
files_list_file_name = None

config_data_g = None




def _read_config(config_data):
    """Reads the relevant for the script information from the configuration dictionary.
    """
    global base_url, download_dir, start_date, end_date, sensor_type, files_list_file_name
    global config_data_g

    config_data_g = config_data
    #Reading config
    base_url = config_data["download_module"]["base_url"]
    print("The base URL set is: " + base_url)
    start_date = config_data["download_module"]["start_date"]
    end_date = config_data["download_module"]["end_date"]
    print("Period to download: " + start_date + "," + end_date)
    sensor_type = config_data["download_module"]["sensor_type"]
    download_dir = os.path.expanduser(config_data["raw_down_dir"])

    files_list_file_name = config_data["download_module"]["files_list_file"]
    files_list_file_name = os.path.expanduser(files_list_file_name)




def _get_all_files_to_downloa(dates):

    if not files_list_file_name is None:
        files_list_file = open(files_list_file_name, "w") 
    else:
        return
    for date in np.nditer(dates):
        day_link =  base_url + "/" + str(date)
        html_on_page = requests.get(day_link).text
        links_on_page = BeautifulSoup(html_on_page, "lxml").findAll("a")
        links_strings = map(
            lambda el: el.get("href"),
            links_on_page)
        links_strings = list(filter(
            lambda el: re.match(".*" + sensor_type + ".*" + ".*.csv", el),
            links_strings))
        for idx,link in enumerate(links_strings):
            files_list_file.write(day_link + "/" + link + "," + download_dir + "/" + link + "\n")
        print("Date written: " + str(date))
    print("Files to be dowloaded are written in the files_list file.")
            

def _download_files():

    print("Donwloading")
    
    save_list = False
    if not files_list_file_name is None:
        files_list_file = open(files_list_file_name, "w") 
        save_list = True
        
    
    dates = np.arange(np.datetime64(start_date), np.datetime64(end_date))
    if "list_files" in config_data_g["download_module"]:
        _get_all_files_to_downloa(dates)

        


    
    print(str(dates.size) + " dates found")
    for date in np.nditer(dates):
        day_link =  base_url + "/" + str(date)
        print("Processing directory: " + day_link)

        html_on_page = requests.get(day_link).text
        links_on_page = BeautifulSoup(html_on_page, "lxml").findAll("a")
        
        links_strings = map(
            lambda el: el.get("href"),
            links_on_page)
        links_strings = list(filter(
            lambda el: re.match(".*" + sensor_type + ".*" + ".*.csv", el),
            links_strings))

        size = str(len(links_strings))
        print(size + " files found")

        for idx,link in enumerate(links_strings):


            if idx % 10 == 0:
                print(str(idx) + "/" + size +" processed")


                
            while True:
                try:
                    wget.download(day_link + "/" + link, out=download_dir)
                    break
                except:
                    continue
            
            print("\n")
            
    if save_list:
        files_list_file.close()


    
    

def main():
    #Basic Setup
    print("Starting downlaod module from command line")
    config_data = ut.get_config_data(sys.argv[1]) 
    print("Configuration file loaded")
    ut.sanity_cahecks(config_data)
    _read_config(config_data)
    _download_files()



if __name__ == '__main__':
    main()
