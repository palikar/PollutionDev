#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal
from edward.models import NormalWithSoftplusScale
from scipy.stats import norm
from matplotlib import pyplot as plt
import utils as ut
import os


class Bnn:
    """
    A class for abstracting the structure of Bayesian Neural Network
    """

    def __init__(self, model_id):
        self.model_id = model_id

        

    def build(self, input_dim, output_dim, layers_defs=[3,3], examples=50):
        """
        Constructs a Tensorflow graph for the the BNN.
        """

        print("Generating prior Variables")
        self.priorWs, self.priorBs = self.generate_prior_vars(input_dim, output_dim, layers_defs)
        print("Generating latent Variables")
        self.qWs, self.qBs = self.generate_latent_vars(input_dim,output_dim, layers_defs)

        
        print("Building Network for inference")
        #Final function of the network

        self.X = tf.placeholder(shape=[None, input_dim], name="input_placeholder", dtype=tf.float32)
        self.y = Normal(
            loc=self._neural_network(self.X, self.priorWs, self.priorBs),
            scale=0.1 * tf.ones(examples)
        )
        self.y_ph = tf.placeholder(tf.float32, self.y.shape, "output_placeholder")

                
        print("Building Network for evaluation")

        self.x_evaluation = tf.placeholder(shape=[None,input_dim], name="evaluation_placeholder", dtype=tf.float32)        
        self.evaluation_sample_count = tf.placeholder(shape=None, name="evaluation_sample_count", dtype=tf.int32)        

        self.y_evaluation = tf.map_fn(
            lambda _: self._neural_network(
                self.x_evaluation,
                list(map(lambda W: W.sample(), self.qWs)),
                list(map(lambda b: b.sample(), self.qBs))
            ) ,
            tf.range(self.evaluation_sample_count),
            dtype=tf.float32)
        self.y_evaluation = tf.identity(self.y_evaluation, name="evaluation")

        self.init_op = tf.global_variables_initializer()
        ed.get_session().run(self.init_op)
        self.saver = tf.train.Saver()

        
    

    def _neural_network(self,x, Ws, bs):
        """
        Constructs a Tensorflow operation for a neural network given weight matrices, biases and input for the network.
        """
        h = tf.tanh(tf.matmul(x, Ws[0]) + bs[0])
        for W, b in zip(Ws[1:-1], bs[1:-1]):
            h = tf.tanh(tf.matmul(h, W) + b)
        h = tf.matmul(h, Ws[-1]) + bs[-1]
        return tf.reshape(h, [-1])


    def generate_prior_vars(self,input_dim, output_dim, layers_defs):

        priorWs=list()
        priorBs=list()


        if len(layers_defs) == 0:
            priorWs.append(Normal(loc=tf.zeros([input_dim, input_dim]), scale=tf.ones([input_dim, input_dim])))
            priorBs.append(Normal(loc=tf.zeros(input_dim), scale=tf.ones(input_dim)))        
        else:
            priorWs.append(Normal(loc=tf.zeros([input_dim, layers_defs[0]]), scale=tf.ones([input_dim, layers_defs[0]])))
            priorBs.append(Normal(loc=tf.zeros(layers_defs[0]), scale=tf.ones(layers_defs[0])))        
        
        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - 1:
                priorWs.append(Normal(loc=tf.zeros([layers_defs[i], layers_defs[i]]),scale=tf.ones([layers_defs[i], layers_defs[i]])))
                priorBs.append(Normal(loc=tf.zeros(layers_defs[i]), scale=tf.ones(layers_defs[i])))
            else:
                priorWs.append(Normal(loc=tf.zeros([layers_defs[i], layers_defs[i+1]]),scale=tf.ones([layers_defs[i], layers_defs[i+1]])))
                priorBs.append(Normal(loc=tf.zeros(layers_defs[i+1]), scale=tf.ones(layers_defs[i+1])))            


        if len(layers_defs) == 0:
            priorWs.append(Normal(loc=tf.zeros([input_dim, output_dim]), scale=tf.ones([input_dim, output_dim])))
            priorBs.append(Normal(loc=tf.zeros(output_dim), scale=tf.ones(output_dim)))
        else:
            priorWs.append(Normal(loc=tf.zeros([layers_defs[-1], output_dim]), scale=tf.ones([layers_defs[-1], output_dim])))
            priorBs.append(Normal(loc=tf.zeros(output_dim), scale=tf.ones(output_dim)))

        return priorWs,priorBs
        
    
    def generate_latent_vars(self,input_dim, output_dim, layers_defs):

       
        qWs=list()
        qBs=list()
        
        if len(layers_defs) == 0:
            qWs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([input_dim, input_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([input_dim, input_dim])))))
            qBs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([input_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([input_dim])))))
        else:
            qWs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([input_dim, layers_defs[0]])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([input_dim, layers_defs[0]])))))
            qBs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[0]])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[0]])))))

        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - 1:
                qWs.append(
                    NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[i], layers_defs[i]])),
                                            scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[i], layers_defs[i]])))))
                qBs.append(
                    NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[i]])),
                                            scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[i]])))))                
            else:
                qWs.append(
                    NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[i], layers_defs[i+1]])),
                                            scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[i], layers_defs[i+1]])))))
                qBs.append(
                    NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[i+1]])),
                                            scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[i+1]])))))

       
        if len(layers_defs) == 0:

            qWs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([input_dim, output_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([input_dim, output_dim])))))
            qBs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([output_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([output_dim])))))

        else:            
            qWs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([layers_defs[-1], output_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([layers_defs[-1], output_dim])))))
            qBs.append(
                NormalWithSoftplusScale(loc=tf.Variable(tf.random_normal([output_dim])),
                                        scale=tf.Variable(tf.Variable(tf.random_normal([output_dim])))))


        return qWs,qBs


    def reset(self):
        """Reinitialized all variables in the constructed graph with random
        values. If the model is trained, callubg this method will
        reset it.

        """
        ed.get_session().run(self.init_op)
    

    def fit(self, X, y, M=None, epochs=1, updates_per_batch=1, samples=30, callback=None):
        """Trains the network with the given in X and y data.
        epochs: The iteration count over the whole dataset
        M: The size of the batch that should be itertated over for optimization
        updates_per_batch: The count of consecetive interations over the same batch
        samples: samples drawn from the mdoels for calucating grading descent
        callback: a function to be called every 1000 epochs while training the model
        """
        latent_vars = {}
        N = y.shape[0]
        for var, q_var in zip(self.priorWs, self.qWs):
            latent_vars[var] = q_var
        for var, q_var in zip(self.priorBs, self.qBs):
            latent_vars[var] = q_var

        if M is None:
            M = N
            
        n_batch = int(N / M)
        n_epoch = epochs
        data = ut.generator([X, y], M)


        inference = ed.KLqp(latent_vars, data={self.y: self.y_ph})        
        inference.initialize(
            n_iter=n_epoch * n_batch * updates_per_batch,
            n_samples=samples,
            scale={self.y: N / M})
        tf.global_variables_initializer().run()
        
        print("Total iterations: " + str(inference.n_iter))

        for i in range(n_epoch):
            total_loss = 0
            for _ in range(inference.n_iter // updates_per_batch // n_epoch):
                X_batch, y_batch = next(data)
                for _ in range(updates_per_batch):
                    info_dict = inference.update({self.y_ph:y_batch, self.X:X_batch})
                total_loss += info_dict['loss']
            
            print("Epoch "+str(i)+" complete. Total loss: " + str(total_loss))
            if i % 1000 == 0 and callback is not None:
                callback(self, i)
                
        


    def evaluate(self, x, samples_count):
        """Draws a samples form the model given some unseen data.
        """
        op = tf.get_default_graph().get_tensor_by_name("evaluation:0")

        x_evaluation = tf.get_default_graph().get_tensor_by_name("evaluation_placeholder:0")
        evaluation_sample_count = tf.get_default_graph().get_tensor_by_name("evaluation_sample_count:0")

        sess=ed.get_session()        
        res = sess.run(op,
            feed_dict={
                x_evaluation : x,
                evaluation_sample_count : samples_count
            }
        )
        
        return res
        

    def save(self, directory, name):
        """Saves the graph and all variables to files on disk. Everything will
        be put in a new direcotry given by the argument.

        """
        directory_exp = os.path.expanduser(directory)
        if not os.path.isdir(directory_exp):
            os.makedirs(directory_exp)            
        self.saver.save(ed.get_session(), os.path.join(directory_exp,name))
    

    def load(self, directory, name):
        """Loads a model from the disk that was saved beforehand.
        """
        sess =  ed.get_session()
        directory_exp = os.path.expanduser(directory)
        self.saver = tf.train.import_meta_graph(directory_exp + name +".meta")
        self.saver.restore(sess, tf.train.latest_checkpoint(directory_exp))

    
        
        
