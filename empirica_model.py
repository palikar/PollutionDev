#!/home/arnaud/anaconda3/bin/python3

import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Empirical


class Emp:
    """A very simple wrapper around the Empirical class of Edward. Once
    the model is built with some samples, one can then draw aditional
    samples from it through the evaluate method

    """
    def __init__(self, model_id):
        self.model_id = model_id        

    def build(self,Y):
        self.var = Empirical(Y)
        return self

    def getVar(self):
        return self.var

    def evaluate(self, num_samples):
        return self.var.sample([num_samples]).eval(session=ed.get_session())





    
