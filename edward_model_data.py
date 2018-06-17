#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from scipy.stats import norm
from edward.models import Normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d



def main():
    "Entry point of cli"


    example_size = 75
    

    
    rand = norm.rvs(size=example_size, loc=0, scale=0.2)


    
 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Grab some test data.
    # X, Y, Z = axes3d.get_test_data(0.05)

    x_train_1 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    x_train_2 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    x_train = np.array([x_train_1, x_train_2])
    
    X = x_train_1
    Y = x_train_2
    X, Y = np.meshgrid(X, Y)

    y_train = np.add(X,Y,dtype=np.float32)
    y_train = np.cos(y_train + rand)
    
    Z = y_train
    
    print(X.shape)
    print("----")
    print(Y.shape)
    print("----")
    print(Z.shape)
    print("----")
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    print(x.shape)
    print("----")
    print(y.shape)
    print("----")
    print(z.shape)
    print("----")
    
    
    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z)
    # ax.plot_surface(x, y, z)

    plt.show()
    
    


if __name__ == '__main__':
    main()

    

