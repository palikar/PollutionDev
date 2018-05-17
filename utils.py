#!/usr/bin/python

import json
import os
import sys



def get_config_data(config_file_name):
    
    with open(config_file_name, "r") as config_file:
        config_data = json.load(config_file)
        return config_data



def sanity_cahecks(config):
    # directory for the data of all scripts
    if not "env_dir" in config:
        print("Envinroment directory not set in the config file (env_dir)")
        sys.exit("Exiting with error")

        if not os.path.isdir(os.path.expanduser(config["env_dir"])):
            os.makedirs(os.path.expanduser(config["env_dir"]))

    if not "env_dir" in config:
        print("Download directory not set in the config file (raw_down_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["raw_down_dir"])):
        os.makedirs(os.path.expanduser(config["raw_down_dir"]))


    if not "data_files_dir" in config:
        print("Final data files directory not set in the config file (data_files_dir)")
        sys.exit("Exiting with error")
    if not os.path.isdir(os.path.expanduser(config["data_files_dir"])):
        os.makedirs(os.path.expanduser(config["data_files_dir"]))




        
if __name__ == '__main__':
    print("This file is not to be executed from the command line")
            
