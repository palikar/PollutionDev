#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal


def neural_network(x, Ws, bs):
    layers = Ws.shape[0]
    h = tf.tanh(tf.matmul(x, W[0]) + b[0])
    for i in range(1, layers):
        h = tf.tanh(tf.matmul(x, W[i]) + b[i])

    return tf.reshape(h, [-1])

class Bnn:


    def __init__(self):
        "Doc"
        self.modeli_id = ""
        self.layers_defs = (None)
        

    def build(self, layers_defs):
        pass

    def fit(self, X, y):
        pass


    def eveluate(self, x):
        "Returns a sampler function n->[n samples from the generated distribuition]"
        pass

    def save(self, direcotry):
        pass

    def load(self, direcotry):
        pass
        
        
        
def main():
    print("Starign BNN")
    


if __name__ == '__main__':
    main()







        
