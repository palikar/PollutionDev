#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal
from edward.models import Gamma
from edward.models import Beta
from edward.models import Exponential
from edward.models import Poisson
from edward.models import Empirical
from scipy.stats import norm
from matplotlib import pyplot as plt
import utils as ut
import os


class Emp:


    def __init__(self, model_id):
        "Doc"
        self.model_id = model_id
        
        

    def build(self,Y):
        self.var = Empirical(Y)
        return self

    def getVar(self):
        return self.var
        
    

    def evaluate(self, num_samples):
        return self.var.sample([num_samples]).eval(session=ed.get_session())
        
    
        
        
        
def main():
    print("Starting Empirical")
    

    example_size = 500
    x_train_1 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    x_train_2 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    rand = norm.rvs(size=example_size, loc=0, scale=0.12)
    x_train_2 = np.linspace(0, 2, num=example_size,dtype=np.float32)

    x_train = np.array([x_train_1, x_train_2]).T


    y_train = np.add(x_train_1,x_train_2,dtype=np.float32)
    y_train = np.sin(np.add(y_train, rand)).T.reshape(example_size,1)
    y_train /= 2

    

    inputs_1 = np.linspace(0,2 , num=example_size, dtype=np.float32)
    inputs_2 = np.linspace(0,2 , num=example_size, dtype=np.float32)
    x = np.array([inputs_1, inputs_2]).T
    

    

    

    

    plt.figure(figsize=(15,13), dpi=100)

    model = Emp("sad")

    prediction = 400

    samples=150

    res = np.array([
        model.build(y_train[0:i]).evaluate(samples).reshape(samples)
        for i in range(400,500)
        
    ], dtype=np.float32)
    print(res.shape)

    mus = res.mean(axis=1)
    std = res.std(axis=1)

    print(mus.shape)
    print(std.shape)
    
    
    plt.plot(np.arange(y_train.shape[0]) ,y_train, '.b' , linewidth=0.3,label='training data')
    plt.plot(np.arange(400, 500) ,mus, '-r', linewidth=1.1, label='mean of posterior')

    plt.fill_between(np.arange(400, 500), np.percentile(res, 5, axis=1), np.percentile(res, 95, axis=1) , color="red", alpha=0.2)

    plt.legend()
    plt.title("Empirical Model")
    plt.xlabel("point[i]")
    plt.ylabel("output")

    plt.show()

    


if __name__ == '__main__':
    main()







    
