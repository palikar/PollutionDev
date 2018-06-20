#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from scipy.stats import norm
from edward.models import Normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d




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
    
    W_0 = Normal(loc=tf.zeros([2, 2]), scale=tf.ones([2, 2]))
    W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
    b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
    b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

    x = x_train
    y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1),
               scale=0.1 * tf.ones(example_size))




    qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [2, 2]),
                  scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [2, 2])))
    qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [2, 1]),
                  scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [2, 1])))
    qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [2]),
                  scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
    qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]),
                  scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))




    saver = tf.train.Saver()
    
    rs = np.random.RandomState(0)
    inputs_1 = np.linspace(0,2 , num=400, dtype=np.float32)
    inputs_2 = np.linspace(0,2 , num=400, dtype=np.float32)
    x = np.array([inputs_1, inputs_2]).T
    mus = tf.stack(
        [neural_network(x, qW_0.sample(), qW_1.sample(),
                        qb_0.sample(), qb_1.sample())
     for _ in range(10)])


    init_op = tf.global_variables_initializer()
    ed.get_session().run(init_op)
    outputs = mus.eval(session=ed.get_session())
    print(outputs)
    print("------------------")

    
    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                         W_1: qW_1, b_1: qb_1}, data={y: y_train})
    
    inference.run(n_iter=1000,n_samples=5)



    print("Saving")
    saver.save(ed.get_session(), 'models/edward_model')
    outputs = mus.eval(session=ed.get_session())
    print(outputs)
    print("------------------")

    
    print("Done")

    
    
    # plt.figure(figsize=(15,13), dpi=100)
    
    # plt.subplot(2,1,1)
    # plt.plot(np.arange(0, len(y_train)),y_train, 'ks', color="blue", linewidth=1.5)
    # plt.title("Edward: Train Data")
    # plt.xlabel("point[i]")
    # plt.ylabel("output")
    

    # plt.subplot(2,1,2)

    # plt.plot(np.linspace(0, len(outputs[0].T), num=len(y_train)),y_train, 'ks', color="blue", linewidth=1.5)
    
    # plt.plot(np.arange(len(outputs[0].T)), outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
    # plt.plot(np.arange(len(outputs[0].T)), outputs[1:].T, 'r', lw=2, alpha=0.5)


    
    # plt.title("Edward: Model")
    # plt.xlabel("point[i]")
    # plt.ylabel("output")

    # plt.savefig("edward_example_bnn.png", bbox_inches='tight')

    
    


    # config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    # sess = tf.Session(config=config)    


    # sess.close()




if __name__ == '__main__':
    main()

    

