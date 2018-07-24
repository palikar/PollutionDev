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


class Bnn:


    def __init__(self, model_id):
        "Doc"
        self.model_id = model_id
        
        

    def build(self, input_dim, output_dim, layers_defs=[3,3], activation=tf.nn.tanh):


        print("Generating prior Variables")
        self.priorWs, self.priorBs = self.generate_prior_vars(input_dim, output_dim, layers_defs)
        print("Generating latent Variables")
        self.qWs, self.qBs = self.generate_latent_vars(input_dim,output_dim, layers_defs)



        
        print("Building Network for inference")
        #Final function of the network
        self.X = tf.placeholder(shape=[None, input_dim], name="input_placeholder", dtype=tf.float32)
        self.y = Normal(
            loc=self._neural_network(self.X,self.priorWs,self.priorBs),
            scale=0.1 
        )
        self.y_ph = tf.placeholder(tf.float32, self.y.shape)

                
        print("Building Network for evaluation")
        self.x_evaluation = tf.placeholder(shape=[None,input_dim], name="evaluation_placeholder", dtype=tf.float32)
        self.evaluation_sample_count = tf.placeholder(shape=None, name="evaluation_sample_count", dtype=tf.int32)        
        self.y_evaluation = tf.map_fn(
            lambda _: self._neural_network(
                self.x_evaluation,
                list(map(lambda x:x.sample(), self.qWs)),
                list(map(lambda x:x.sample(), self.qBs))) ,
            tf.range(self.evaluation_sample_count),
            dtype=tf.float32)

        
        
        

        self.init_op = tf.global_variables_initializer()
        ed.get_session().run(self.init_op)
        self.saver = tf.train.Saver()
        

        

    def _neural_network(self,x, Ws, bs):
        h = tf.tanh(tf.matmul(x, Ws[0]) + bs[0])
        for W, b in zip(Ws[1:], bs[1:]):
            h = tf.tanh(tf.matmul(h, W) + b)
        return h


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
            qWs.append(Normal(loc=tf.get_variable("q_W_input/loc",[input_dim, input_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_W_input/scale",[input_dim, input_dim]))))
            qBs.append(Normal(loc=tf.get_variable("q_B_input/loc", [input_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_B_input/scale", [input_dim]))))

        else:
            qWs.append(Normal(loc=tf.get_variable("q_W_input/loc",[input_dim, layers_defs[0]]),
                              scale=tf.nn.softplus(tf.get_variable("q_W_input/scale",[input_dim, layers_defs[0] ]))))
            qBs.append(Normal(loc=tf.get_variable("q_B_input/loc", [layers_defs[0]]),
                              scale=tf.nn.softplus(tf.get_variable("q_B_input/scale", [layers_defs[0]]))))

        
        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - 1:
                qWs.append(Normal(loc=tf.get_variable("q_W_"+str(i)+"/loc",[layers_defs[i], layers_defs[i]]),
                                  scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale",[layers_defs[i], layers_defs[i]]))))
                qBs.append(Normal(loc=tf.get_variable("q_B_"+str(i)+"/loc", [layers_defs[i]]),
                                  scale=tf.nn.softplus(tf.get_variable("q_B_"+str(i)+"/scale", [layers_defs[i]]))))

            else:
                qWs.append(Normal(loc=tf.get_variable("q_W_"+str(i)+"/loc",[layers_defs[i], layers_defs[i+1]]),
                                  scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale",[layers_defs[i], layers_defs[i+1]]))))
                qBs.append(Normal(loc=tf.get_variable("q_B_"+str(i)+"/loc", [layers_defs[i+1]]),
                                  scale=tf.nn.softplus(tf.get_variable("q_B_"+str(i)+"/scale",[layers_defs[i+1]]))))                


       
        if len(layers_defs) == 0:
            qWs.append(Normal(loc=tf.get_variable("q_W_output/loc",[input_dim, output_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_W_output/scale",[input_dim, output_dim]))))
            qBs.append(Normal(loc=tf.get_variable("q_B_output/loc", [output_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_B_output/scale",[output_dim]))))
        else:
            qWs.append(Normal(loc=tf.get_variable("q_W_output/loc",[layers_defs[-1], output_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_W_output/scale",[layers_defs[-1], output_dim]))))
            qBs.append(Normal(loc=tf.get_variable("q_B_output/loc", [output_dim]),
                              scale=tf.nn.softplus(tf.get_variable("q_B_output/scale",[output_dim]))))

        return qWs,qBs


    def reset(self):
        ed.get_sessxion().run(self.init_op)

    
    def fit(self, X, y, M=1, epochs=1, updates_per_batch=1):
        latent_vars = {}
        N = y.shape[0]
        for var, q_var in zip(self.priorWs, self.qWs):
            latent_vars[var] = q_var
        for var, q_var in zip(self.priorBs, self.qBs):
            latent_vars[var] = q_var

        
        n_batch = int(N / M)
        n_epoch = epochs
        data = ut.generator([X, y], M)


        inference = ed.KLqp(latent_vars, data={self.y: self.y_ph})
        
        inference.initialize(
            n_iter=n_epoch * n_batch * updates_per_batch,
            n_samples=5,
            scale={self.y: N / M})
        tf.global_variables_initializer().run()
        print(str(inference.n_iter))
        for i in range(n_epoch):
            total_loss = 0
            for _ in range(inference.n_iter // updates_per_batch //n_epoch):
                X_batch, y_batch = next(data)
                for _ in range(updates_per_batch):
                    info_dict = inference.update({self.y_ph:y_batch, self.X:X_batch})
                total_loss += info_dict['loss']
            print("Epoch "+str(i)+" complete. Total loss: " + str(total_loss),  end="\r")
        


    def evaluate(self, x, samples_count):

        res = self.y_evaluation.eval(
            feed_dict={
                self.x_evaluation : x,
                self.evaluation_sample_count : samples_count
            },
            session=ed.get_session()
        )
        
        return res
        

    def save(self, directory, name):
        directory_exp = os.path.expanduser(directory)
        if not os.path.isdir(directory_exp):
            os.makedirs(directory_exp)            
        self.saver.save(ed.get_session(), os.path.join(directory_exp,name))
    

    def load(self, directory, name):
        sess =  ed.get_session()
        directory_exp = os.path.expanduser(directory)
        print(directory_exp + name)
        self.saver = tf.train.import_meta_graph(directory_exp + name +".meta")
        self.saver.restore(sess, tf.train.latest_checkpoint(directory_exp))

    
        
        
        
def main():
    print("Starting BNN")
    
    model = Bnn("bnn_3_3")
    model.build(2,1,layers_defs=[20,20])

    example_size = 50
    x_train_1 = np.linspace(0, 3, num=example_size,dtype=np.float32)
    x_train_2 = np.linspace(0, 3, num=example_size,dtype=np.float32)
    


    x_train = np.array([x_train_1, x_train_2]).T

    rand = norm.rvs(size=example_size, loc=0, scale=0.12)
    y_train = np.add(x_train_1,x_train_2,dtype=np.float32)
    y_train = np.sin(np.add(y_train, rand)).T.reshape(example_size,1)



    model.fit(x_train, y_train, M=example_size, updates_per_batch=2, epochs=10000)
    
    
    
    samples=100
    outputs = model.evaluate(x_train, samples)

    
    plt.figure(figsize=(15,13), dpi=100)
    
    outputs = outputs.reshape(samples,example_size)
    line, = plt.plot(np.arange(len(outputs[0].reshape(-1))), np.mean(outputs, 0).reshape(-1),'r', lw=2, label="posterior mean")

    plt.fill_between(np.arange(len(outputs[0].reshape(-1))),
                     np.percentile(outputs, 5, axis=0),
                     np.percentile(outputs, 95, axis=0),
                     color=line.get_color(), alpha = 0.3, label="confidence_region")
    
    plt.plot(np.linspace(0, len(outputs[0].reshape(-1)), num=len(y_train)), y_train, '.b', color="blue", linewidth=0.3,label='training data')
    
    
    plt.legend()
    plt.title("Edward: Model")
    plt.xlabel("point[i]")
    plt.ylabel("output")

    plt.show()

    


if __name__ == '__main__':
    main()







    
