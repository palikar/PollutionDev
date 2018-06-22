#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal
from scipy.stats import norm
from matplotlib import pyplot as plt
import utils as ut



class Bnn:


    def __init__(self, model_id):
        "Doc"
        self.model_id = model_id
        
        

    def build(self, input_dim, output_dim, example_size, layers_defs=[3,3], activation=tf.nn.tanh,):


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
            qWs.append(Normal(loc=tf.get_variable("q_W_input/loc",initializer=tf.random_uniform([input_dim, input_dim])),
                              scale=tf.nn.softplus(tf.get_variable("q_W_input/scale", initializer=tf.random_uniform([input_dim, input_dim],minval=-10,maxval=10)))))
            qBs.append(Normal(loc=tf.get_variable("q_B_input/loc", initializer=tf.random_uniform([input_dim])),
                              scale=tf.nn.softplus(tf.get_variable("q_B_input/scale", initializer=tf.random_uniform([input_dim],minval=-10,maxval=10)))))
        else:
            qWs.append(Normal(loc=tf.get_variable("q_W_input/loc", initializer=tf.random_uniform([input_dim, layers_defs[0]],minval=-10,maxval=10)),
                              scale=tf.nn.softplus(tf.get_variable("q_W_input/scale", initializer=tf.random_uniform([input_dim, layers_defs[0]])))))
            qBs.append(Normal(loc=tf.get_variable("q_B_input/loc", initializer=tf.random_uniform([layers_defs[0]])),
                              scale=tf.nn.softplus(tf.get_variable("q_B_input/scale", initializer=tf.random_uniform([layers_defs[0]])))))
        
        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - 1:
                qWs.append(Normal(loc=tf.get_variable("q_W_"+str(i)+"/loc", initializer=tf.random_uniform([layers_defs[i], layers_defs[i]],minval=-10,maxval=10)),
                                    scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale", initializer=tf.random_uniform([layers_defs[i], layers_defs[i]],minval=-10,maxval=10)))))
                qBs.append(Normal(loc=tf.get_variable("q_B_"+str(i)+"/loc", initializer=tf.random_uniform([layers_defs[i]],minval=-10,maxval=10)),
                                    scale=tf.nn.softplus(tf.get_variable("q_B_"+str(i)+"/scale", initializer=tf.random_uniform([layers_defs[i]],minval=-10,maxval=10)))))

            else:
                qWs.append(Normal(loc=tf.get_variable("q_W_"+str(i)+"/loc", initializer=tf.random_uniform([layers_defs[i], layers_defs[i+1]],minval=-10,maxval=10)),
                                    scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale", initializer=tf.random_uniform([layers_defs[i], layers_defs[i+1]],minval=-10,maxval=10)))))
                qBs.append(Normal(loc=tf.get_variable("q_B_"+str(i)+"/loc", initializer=tf.random_uniform([layers_defs[i+1]],minval=-10,maxval=10)),
                                    scale=tf.nn.softplus(tf.get_variable("q_B_"+str(i)+"/scale", initializer=tf.random_uniform([layers_defs[i+1]],minval=-10,maxval=10)))))               


        
        if len(layers_defs) == 0:
            qWs.append(Normal(loc=tf.get_variable("q_W_output/loc", initializer=tf.random_uniform([input_dim, output_dim],minval=-10,maxval=10)),
                                scale=tf.nn.softplus(tf.get_variable("q_W_output/scale", initializer=tf.random_uniform([input_dim, output_dim],minval=-10,maxval=10)))))
            qBs.append(Normal(loc=tf.get_variable("q_B_output/loc", initializer=tf.random_uniform([output_dim])),
                          scale=tf.nn.softplus(tf.get_variable("q_B_output/scale", initializer=tf.random_uniform([output_dim])))))
        else:
            qWs.append(Normal(loc=tf.get_variable("q_W_output/loc", initializer=tf.random_uniform([layers_defs[-1], output_dim],minval=-10,maxval=10)),
                                scale=tf.nn.softplus(tf.get_variable("q_W_output/scale", initializer=tf.random_uniform([layers_defs[-1], output_dim])))))
            qBs.append(Normal(loc=tf.get_variable("q_B_output/loc", initializer=tf.random_uniform([output_dim])),
                          scale=tf.nn.softplus(tf.get_variable("q_B_output/scale", initializer=tf.random_uniform([output_dim],minval=-10,maxval=10)))))

        return qWs,qBs


    def reset(self):
        ed.get_session().run(self.init_op)

    
    def fit(self, X, y, n_iter=1000, M=1, epochs=1, updates_per_batch=10):
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
                    inference.print_progress(info_dict)
                total_loss += info_dict['loss']

            print("\nEpoch "+str(i)+" complete. Total loss: " + str(total_loss))
        


    def evaluate(self, x, samples_count):

        res = self.y_evaluation.eval(
            feed_dict={
                self.x_evaluation : x,
                self.evaluation_sample_count : samples_count
            },
            session=ed.get_session()
        )
        
        return res
        

    def save(self, direcotry):
        pass

    def load(self, direcotry):
        pass

        
        
        
def main():
    print("Starting BNN")

    model = Bnn("bnn_3_3")
    model.build(2,1,100,layers_defs=[1])

    example_size = 40
    x_train_1 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    x_train_2 = np.linspace(0, 2, num=example_size,dtype=np.float32)
    rand = norm.rvs(size=example_size, loc=0, scale=0.001)
    x_train = np.array([x_train_1, x_train_2]).T
    y_train = np.multiply(x_train_1,x_train_2,dtype=np.float32)
    y_train = np.sin(np.add(y_train, rand)).T.reshape(example_size,1)



    model.fit(x_train, y_train, n_iter=20000, M=40, updates_per_batch=3, epochs=5000)

    inputs_1 = np.linspace(0,2 , num=100, dtype=np.float32)
    inputs_2 = np.linspace(0,2 , num=100, dtype=np.float32)
    x = np.array([inputs_1, inputs_2]).T
    
    # outputs = model.evaluate(np.array([[1,3],[1,3]]),10)
    outputs = model.evaluate(x,10)

    # print(outputs[0].reshape(-1))
    
    plt.figure(figsize=(15,13), dpi=100)
    
    plt.plot(np.linspace(0, len(outputs[0].reshape(-1)), num=len(y_train)) ,y_train, 'ks', color="blue", linewidth=1.5,label='training data')

    for i in range(10):
        plt.plot(np.arange(len(outputs[i].reshape(-1))), outputs[i].reshape(-1), 'r', lw=2, alpha=0.5)
        
    # plt.plot(np.arange(len(outputs[0].T)), outputs[1:].T, 'r', lw=2, alpha=0.5)

    
    plt.legend()
    plt.title("Edward: Model")
    plt.xlabel("point[i]")
    plt.ylabel("output")

    plt.show()

    


if __name__ == '__main__':
    main()







        
