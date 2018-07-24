#!/usr/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bnn_model import *
from gm_model import *
from empirica_model import *


import utils as ut
import edward as ed
from scipy.stats import norm
import tensorflow as tf

from sklearn import model_selection

import os, sys


# df_12H = pd.read_csv("./env/data_frames/final_data_frame_12H.csv", sep=";", index_col="timestamp", parse_dates=True)
# df_1H = pd.read_csv("./env/data_frames/final_data_frame_1H.csv", sep=";", index_col="timestamp", parse_dates=True)
df_1D = pd.read_csv("./env/data_frames/final_data_frame_1D.csv", sep=";", index_col="timestamp", parse_dates=True)


P1_colmns_1D = list(filter(lambda col: "P1" in str(col),list(df_1D.columns.values)))
P2_colmns_1D = list(filter(lambda col: "P2" in str(col),list(df_1D.columns.values)))

# P1_colmns_12H = list(filter(lambda col: "P1" in str(col),list(df_12H.columns.values)))
# P2_colmns_12H = list(filter(lambda col: "P2" in str(col),list(df_12H.columns.values)))

# P1_colmns_1H= list(filter(lambda col: "P1" in str(col),list(df_1H.columns.values)))
# P2_colmns_1H = list(filter(lambda col: "P2" in str(col),list(df_1H.columns.values)))



sensors = np.array([P1_colmns_1D, P2_colmns_1D])[0]
print(sensors)
ut.list_coordinates(sensors)






# def test_train_split(X, y, train_size=0.75, random=False):
#     if random:
#         return train_test_split(X, y, train_size=train_size, random_state=42)
#     else:
#         train_cnt = int(round(X.shape[0]*0.75, 0))
#         return X[0:train_cnt], X[train_cnt:], y[0:train_cnt], y[train_cnt:]
    

# def select_data(station, value, period):
#     df = None
#     if period == "1D":
#         df = df_1D
#     elif period == "12H":
#         df = df_12H
#     elif period == "1H":
#         df = df_1H

#     X, y = None, None
#     if value == "P1":
#         columns = list(filter(lambda col: "P1" in str(col),list(df.columns.values)))
#     elif value == "P2":
#         columns = list(filter(lambda col: "P2" in str(col),list(df.columns.values)))
#     else:
#         columns = list(filter(lambda col: "P2" in str(col) or "P1" in str(col),list(df.columns.values)))
        
#     out_col = None
#     if station == "SBC":
#         out_col = -1
#     else:
#         out_col = -2

#     y = df[columns[out_col]].values
#     X = df[columns[0:-3]].values
#     return X, y


# station = "SBC"
# value="P1"
# period="1D"
# train_per = 0.75


# X, y = select_data(station, value, period)
# X_train, X_test, y_train, y_test = test_train_split(X, y,train_size=train_per, random=False)
# y_train = y_train.reshape(y_train.shape[0],1)
# y_test = y_test.reshape(y_test.shape[0],1)



# print(X_train.shape, X_test.shape)
# print(y_train.shape, y_test.shape)


# neurons_per_inner_bnn = 20
# inner_layer_cnt_bnn = 2

# layers = []
# for i in range(inner_layer_cnt_bnn):
#     layers.append(neurons_per_inner_bnn)

# layers = [200,200]

# model_id = "bnn_n"+str(neurons_per_inner_bnn)+"_l"+str(inner_layer_cnt_bnn)+"_s"+str(station)+"_tts" + str(train_per) + "_v" + value + "_p" + period



# print("Model id:" + model_id)
# print("Layers:" + str(layers))

# model = Bnn(model_id)
# model.build(X_train.shape[1],1, layers)


# batch_size_bnn = y_train.shape[0]
# epochs_bnn = 100000
# updates_per_batch_bnn = 1

# model.fit(X_train, y_train, M=batch_size_bnn, updates_per_batch=updates_per_batch_bnn, epochs=epochs_bnn)

# print("Fitting is now complete")

# plt.figure(figsize=(15,13), dpi=100)
# plt.title("Bnn Model")
# plt.xlabel("point[i], t")
# plt.ylabel("output")
    
# samples = 100
# outputs = model.evaluate(X_train, samples)
# outputs = outputs.reshape(samples,X_train.shape[0])

# line, = plt.plot(np.arange(len(outputs[0].reshape(-1))), np.mean(outputs, 0).reshape(-1),'r', lw=2, label="posterior mean")

# plt.fill_between(np.arange(len(outputs[0].reshape(-1))),
#                 np.percentile(outputs, 5, axis=0),
#                 np.percentile(outputs, 95, axis=0),
#                 color=line.get_color(), alpha = 0.3, label="confidence_region")
    
# plt.plot(np.arange(y_train.shape[0]), y_train, '-b' , linewidth=0.5,label='Data')


# plt.legend()
# plt.show()









