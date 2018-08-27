#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
from scipy.stats import norm

import os, sys

import gpflow
from gpflow.saver import Saver
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch, DataHolder
from gpflow.params import Parameter, ParamList
from gpflow.training import AdamOptimizer, ScipyOptimizer
from gpflow.decors import params_as_tensors, autoflow

from scipy.stats import norm
import numpy as np




from matplotlib import pyplot as plt

float_type = gpflow.settings.float_type

class Mdn(Model):
    
    def __init__(self,model_id, X, Y, inner_dims=[10, 10,], activation=tf.nn.tanh, num_mixtures=5, model_file=None ):
        Model.__init__(self)
        self.model_id = model_id

        self.Din = X.shape[1]
        self.dims = [self.Din, ] + list(inner_dims) + [3 * num_mixtures]
        self.activation = activation
        

        self.X = DataHolder(X)
        self.Y = DataHolder(Y)

        self.saver = Saver()

        if model_file is None:
            self._create_network()
        else:
            self.load(model_file)


        
    def _create_network(self):
        Ws, bs = [], []
        for dim_in, dim_out in zip(self.dims[:-1], self.dims[1:]):
            init_xavier_std = (2.0 / (dim_in + dim_out)) ** 0.5
            Ws.append(Parameter(np.random.randn(dim_in, dim_out) * init_xavier_std))
            
            bias_always_postive = Parameter(np.zeros(dim_out), transform=gpflow.transforms.positive)
            bs.append(bias_always_postive)

        self.Ws, self.bs = ParamList(Ws), ParamList(bs)

    @params_as_tensors
    def _eval_network(self, X):
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            X = tf.matmul(X, W) + b
            if i < len(self.bs) - 1:
                X = self.activation(X)

        pis, mus, sigmas = tf.split(X, 3, axis=1)
        pis = tf.nn.softmax(pis)  # make sure they normalize to 1
        sigmas = tf.exp(sigmas)   # make sure std. dev. are positive
        
        return pis, mus, sigmas

    @params_as_tensors
    def _build_likelihood(self):
        pis, mus, sigmas = self._eval_network(self.X)
        Z = (2 * np.pi)**0.5 * sigmas
        log_probs_mog = (-0.5 * (mus - self.Y)**2 / sigmas**2) - tf.log(Z) + tf.log(pis)
        log_probs = tf.reduce_logsumexp(log_probs_mog, axis=1)
        return tf.reduce_sum(log_probs)
    
    @autoflow((float_type, [None, None]))
    def eval_network(self, X):
        pis, mus, sigmas = self._eval_network(X)
        return pis, mus, sigmas


    def save(self, directory):
        directory_exp = os.path.expanduser(directory)
        if not os.path.isdir(directory_exp):
            os.makedirs(directory_exp)            

        self.saver.save(directory_exp+"/"+"model", [self.Ws, self.bs])
            

    def load(self, path):
        data = self.saver.load(path)
        self.Ws = data[0]
        self.bs = data[1]




    def fit(self, num_iter=10000, callback=None):
        self.compile()
        opt = ScipyOptimizer()

        iter_pro_stage = int(num_iter/3)
        
        self.Ws.set_trainable(True)
        self.bs.set_trainable(False)
        opt.minimize(self, maxiter=iter_pro_stage, disp=True)
        if callback is not None:
            callback(self, int(num_iter/3)*1)
        
        self.Ws.set_trainable(False)
        self.bs.set_trainable(True)
        opt.minimize(self, maxiter=iter_pro_stage, disp=True)
        if callback is not None:
            callback(self, int(num_iter/3)*2)

        

        self.Ws.set_trainable(True)
        self.bs.set_trainable(True)
        opt.minimize(self, maxiter=iter_pro_stage, disp=True)
        if callback is not None:
            callback(self, int(num_iter))

