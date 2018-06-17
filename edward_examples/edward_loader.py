#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from scipy.stats import norm
from edward.models import Normal
from matplotlib import pyplot as plt




def neural_network(x, W_0, W_1, b_0, b_1):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])

def main():
    "Entry point of cli"


    example_size = 100
    x_train_1 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    x_train_2 = np.linspace(0, 2, num=example_size,dtype=np.float32)

    
    rand = norm.rvs(size=example_size, loc=0, scale=0.1)


    
    x_train = np.array([x_train_1, x_train_2]).T
    y_train = np.add(x_train_1,x_train_2,dtype=np.float32)
    y_train = np.sin(np.add(y_train, rand)).T

    print(x_train.shape)
    print(y_train.shape)

    outputs = None
    sess =  ed.get_session()

    new_saver = tf.train.import_meta_graph('./models/edward_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    graph = tf.get_default_graph()
    mus = graph.get_tensor_by_name("stack:0")
    outputs = mus.eval()
    print("Done")
                

    print(outputs)
        

    
    
    plt.figure(figsize=(15,13), dpi=100)
    
    plt.subplot(2,1,1)
    plt.plot(np.arange(0, len(y_train)),y_train, 'ks', color="blue", linewidth=1.5)
    plt.title("Edward: Train Data")
    plt.xlabel("point[i]")
    plt.ylabel("output")
    

    plt.subplot(2,1,2)

    plt.plot(np.linspace(0, len(outputs[0].T), num=len(y_train)),y_train, 'ks', color="blue", linewidth=1.5)
    
    plt.plot(np.arange(len(outputs[0].T)), outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
    plt.plot(np.arange(len(outputs[0].T)), outputs[1:].T, 'r', lw=2, alpha=0.5)


    
    plt.title("Edward: Model")
    plt.xlabel("point[i]")
    plt.ylabel("output")

    plt.show()
    # plt.savefig("edward_example_bnn.png", bbox_inches='tight')

    
    


    # config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    # sess = tf.Session(config=config)    


    # sess.close()




if __name__ == '__main__':
    main()

    

