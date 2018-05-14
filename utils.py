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
