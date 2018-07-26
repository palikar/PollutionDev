#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal
from edward.models import Gamma
from edward.models import Beta
from edward.models import Exponential
from edward.models import Poisson
from scipy.stats import norm
from matplotlib import pyplot as plt
import utils as ut
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import normalize

    
        

def build_toy_dataset(N=50, noise_std=0.1, till=2):
    x_train_1 = np.linspace(0, till, num=N,dtype=np.float32)
    x_train_2 = np.linspace(0, till, num=N,dtype=np.float32)
    x_train = np.array([x_train_1, x_train_2]).T

    rand = norm.rvs(size=N, loc=0, scale=0.12)
    y_train = np.add(x_train_1,x_train_2,dtype=np.float32)
    y_train = np.sin(np.add(y_train, rand)).T

    
    return x_train, y_train


def neural_network(x,
                   W_0, W_1, W_2, W_3,
                   b_0, b_1, b_2, b_3):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.tanh(tf.matmul(h, W_2) + b_2)
    h = tf.matmul(h, W_3) + b_3
    return tf.reshape(h, [-1])


        
        
def main():
    print("Starting BNN")
    # ed.set_seed(42)

    N = 50  
    N_eval = 100
    D = 2

    x_train, y_train = build_toy_dataset(N)

    print(x_train)    
    print(x_train.shape)
    print(y_train.shape)

    hid=10

    W_0 = Normal(loc=tf.zeros([D, hid]), scale=tf.ones([D, hid]))
    W_1 = Normal(loc=tf.zeros([hid, hid]), scale=tf.ones([hid, hid]))

    W_2 = Normal(loc=tf.zeros([hid, hid]), scale=tf.ones([hid, hid]))
    W_3 = Normal(loc=tf.zeros([hid, 1]), scale=tf.ones([hid, 1]))
    
    b_0 = Normal(loc=tf.zeros(hid), scale=tf.ones(hid))
    b_1 = Normal(loc=tf.zeros(hid), scale=tf.ones(hid))

    b_2 = Normal(loc=tf.zeros(hid), scale=tf.ones(hid))
    b_3 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

    x = x_train
    y = Normal(loc=neural_network(x,
                                  W_0, W_1, W_2, W_3,
                                  b_0, b_1, b_2, b_3),
               scale=0.1 * tf.ones(N))


    qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [D, hid]), scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [D, hid])))
    qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [hid, hid]), scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [hid, hid])))

    qW_2 = Normal(loc=tf.get_variable("qW_2/loc", [hid, hid]), scale=tf.nn.softplus(tf.get_variable("qW_2/scale", [hid, hid])))
    qW_3 = Normal(loc=tf.get_variable("qW_3/loc", [hid, 1]), scale=tf.nn.softplus(tf.get_variable("qW_3/scale", [hid, 1])))

    qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [hid]), scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [hid])))
    qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [hid]), scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [hid])))

    qb_2 = Normal(loc=tf.get_variable("qb_2/loc", [hid]), scale=tf.nn.softplus(tf.get_variable("qb_2/scale", [hid])))
    qb_3 = Normal(loc=tf.get_variable("qb_3/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb_3/scale", [1])))





    (x, _) = build_toy_dataset(N=N_eval)
    mus = tf.stack(
        [neural_network(x,
                        qW_0.sample(), qW_1.sample(), qW_2.sample(), qW_3.sample(),
                        qb_0.sample(), qb_1.sample(), qb_2.sample(), qb_3.sample())
     for _ in range(20)])

    

    
    sess = ed.get_session()
    tf.global_variables_initializer().run()
    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1,
                         W_2: qW_2, b_2: qb_2,
                         W_3: qW_3, b_3: qb_3},
                        data={y: y_train})
    inference.run(n_iter=10000, n_samples=20)
    outputs = mus.eval()



      
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Iteration: 20000")

    ax.plot(np.linspace(0, N_eval, N), y_train, '.b', alpha=0.5, label='(x, y)')

    line, = ax.plot(np.arange(0, x.shape[0]), np.mean(outputs, axis=0), '-r', lw=2, alpha=0.5, label='mean')

    plt.fill_between(np.arange(0, x.shape[0]),
                     np.percentile(outputs, 5, axis=0),
                     np.percentile(outputs, 95, axis=0),
                     color=line.get_color(), alpha = 0.3, label="confidence_region")

    
    ax.legend()
    plt.show()


    

if __name__ == '__main__':
    main()







    
