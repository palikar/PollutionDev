#!/home/arnaud/anaconda3/bin/python3

import os, sys, argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
import json, re
from scipy import stats
from collections import defaultdict



def change_plot(folds, dest):



    for folder in folds:

        plt.figure(figsize=(10,12), dpi=100)

        after_file = os.path.join(folder, "env/description_frame_p1_after.csv")
        before_file = os.path.join(folder, "env/description_frame_p1_before.csv")

        after_df = pd.read_csv(after_file, sep=';')
        before_df = pd.read_csv(before_file, sep=';')

        n_groups = before_df["id"].values.shape[0]
        index = np.arange(n_groups)
        bar_width = 0.2
        opacity = 0.75

        

        ax = plt.subplot(4,2,1)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["min"].values),
                bar_width,
                alpha=opacity,
                label="Min value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["min"].values),
                bar_width,
                alpha=opacity,
                label="Min value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Min value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Min value comparison of P1")        
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["min"].values, after_df["min"].values))*1.2])
        plt.legend()

        
        ax = plt.subplot(4,2,3)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["max"].values),
                bar_width,
                alpha=opacity,
                label="Max value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["max"].values),
                bar_width,
                alpha=opacity,
                label="Max value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Max value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Max value comparison of P1")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["max"].values, after_df["max"].values))*1.2])
        plt.legend()

                
        ax = plt.subplot(4,2,5)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["mean"].values),
                bar_width,
                alpha=opacity,
                label="Mean value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["mean"].values),
                bar_width,
                alpha=opacity,
                label="Mean value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Mean value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Mean value comparison of P1")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["mean"].values, after_df["mean"].values))*1.2])
        plt.legend()

        ax = plt.subplot(4,2,7)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["std"].values),
                bar_width,
                alpha=opacity,
                label="Std before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["std"].values),
                bar_width,
                alpha=opacity,
                label="Std after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Std value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Std comparison of P1")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["std"].values, after_df["std"].values))*1.2])
        plt.legend()


        
        after_file = os.path.join(folder, "env/description_frame_p2_after.csv")
        before_file = os.path.join(folder, "env/description_frame_p2_before.csv")

        after_df = pd.read_csv(after_file, sep=';')
        before_df = pd.read_csv(before_file, sep=';')

        n_groups = before_df["id"].values.shape[0]
        index = np.arange(n_groups)
        bar_width = 0.2
        opacity = 0.75

        

        ax = plt.subplot(4,2,2)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["min"].values),
                bar_width,
                alpha=opacity,
                label="Min value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["min"].values),
                bar_width,
                alpha=opacity,
                label="Min value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Min value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Min value comparison of P2")        
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["min"].values, after_df["min"].values))*1.2])
        plt.legend()

        
        ax = plt.subplot(4,2,4)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["max"].values),
                bar_width,
                alpha=opacity,
                label="Max value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["max"].values),
                bar_width,
                alpha=opacity,
                label="Max value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Max value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Max value comparison of P2")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["max"].values, after_df["max"].values))*1.2])
        plt.legend()

                
        ax = plt.subplot(4,2,6)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["mean"].values),
                bar_width,
                alpha=opacity,
                label="Mean value before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["mean"].values),
                bar_width,
                alpha=opacity,
                label="Mean value after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Mean value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Mean value comparison of P2")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["mean"].values, after_df["mean"].values))*1.2])
        plt.legend()

        ax = plt.subplot(4,2,8)
        plt.bar(index + 0*bar_width, 
                np.array(before_df["std"].values),
                bar_width,
                alpha=opacity,
                label="Std before preprocessing")
        plt.bar(index + 1*bar_width, 
                np.array(after_df["std"].values),
                bar_width,
                alpha=opacity,
                label="Std after preprocessing")
        plt.xticks(rotation=45)
        plt.xticks(index + bar_width, after_df["id"].values)
        plt.ylabel("Std value")
        plt.xlabel('Sensor ID', fontsize=9)
        plt.title("Std comparison of P2")
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(np.append(before_df["std"].values, after_df["std"].values))*1.2])
        plt.legend()



        
        plt.tight_layout()
        plt.savefig(os.path.join(dest, re.search("data_files_(\w*)",folder).group(1) + "_sensor_metrics.png"))
                     


def corr_plot(folds, dest):
    reg = "Description after[\s\S]*P1_(\w*)\s*1\.0+\s*(\d\.\d*)"


    for folder in folds:

        sens_descs = [os.path.join(folder + "/env/desc_files/", f) for f in os.listdir(folder+"/env/desc_files/")
                       if os.path.isfile(os.path.join(folder+"/env/desc_files/", f))]

        res = dict()
        for desc in sens_descs:
            with open(desc, "r") as desc_file:
                cont = desc_file.read()
                match = re.search(reg, cont)
                # print(match.group(1) + "-" + match.group(2))
                res[match.group(1)] = round(float(match.group(2)),3)
                
        
        n_groups = len(res.keys())
        index = np.arange(n_groups)
        bar_width = 0.5
        opacity = 0.75
        
        plt.figure(figsize=(8,4), dpi=100)
        ax = plt.subplot(1,1,1)
        plt.bar(index + 0*bar_width, 
                res.values(),
                bar_width,
                alpha=opacity,
                label="")
        
        plt.xticks(rotation=45)
        plt.xticks(index + 0*bar_width, res.keys())


        mean = np.mean(list(res.values()))
        mean = round(mean, 3)
        
        plt.ylabel("Correlation between PM2.5 and PM10")
        plt.xlabel('Sensor ID', fontsize=10)
        plt.title("Correlation between PM2.5 and PM10. Mean over all values: " + str(mean))
        plt.xlim(min(index)-bar_width*2, max(index)+bar_width*2)
        plt.ylim([0, max(res.values())*1.2])
        # plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(dest, re.search("data_files_(\w*)",folder).group(1) + "_P1_P2_correlation.png"))
        # plt.show()
        

        

        
        
        pass
    


def main():
    
    parser = argparse.ArgumentParser(description='Generate result plots')
    parser.add_argument('folders', metavar='FOLDs',
                        help='The main folders to process', nargs='+')
    parser.add_argument('--dest', dest='dest', action='store', default="./",
                        help='Folder for the results', required=False)
    
    args = parser.parse_args()

    if not os.path.isdir(args.dest):
        os.makedirs(args.dest)
    

    print("Processing fodlers: ", args.folders)

    print("Generating changes plots")
    change_plot(args.folders, args.dest)
    print("Generating correlation plots")
    corr_plot(args.folders, args.dest)


if __name__ == '__main__':
    main()
