#!/home/arnaud/anaconda3/bin/python3


import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal


class Bnn:


    def __init__(self):
        "Doc"
        self.model_id = ""
        self.layers_defs = (None)
        

    def build(self, input_dim, output_dim, example_size, layers_defs=[3,3], activation=tf.nn.tanh,):


        self.x_input = tf.placeholder(shape=[input_dim], name="input_placeholder")


        priorWs = np.array([])
        priorBs = np.array([])

        priorWs = np.append(priorWs,
                            Normal(loc=tf.zeros([input_dim, layers_defs[0]]), scale=tf.ones([input_dim, layers_defs[0]])))
        priorBs = np.append(priorBs,
                            loc=tf.zeros(input_dim), scale=tf.ones(input_dim))

        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - -1:
                priorWs = np.append(priorWs,
                    Normal(
                        loc=tf.zeros([layers_defs[i], layers_defs[i]]),
                        scale=tf.ones([layers_defs[i], layers_defs[i]]))
                )
                priorBs = np.append(priorBs, loc=tf.zeros(layers_defs[i]), scale=tf.ones(layers_defs[i]))
            else:
                priorWs = np.append(priorWs,
                    Normal(
                        loc=tf.zeros([layers_defs[i], layers_defs[i+1]]),
                        scale=tf.ones([layers_defs[i], layers_defs[i+1]]))
                )
                priorBs = np.append(priorBs, loc=tf.zeros(layers_defs[i]), scale=tf.ones(layers_defs[i]))
        
        priorWs = np.append(priorWs,
                            Normal(loc=tf.zeros([layers_defs[-1], output_dim]), scale=tf.ones([layers_defs[-1], output_dim])))
        priorBs = np.append(priorBs,
                            loc=tf.zeros(layers_defs[-1]), scale=tf.ones(layers_defs[-1]))


        qWs = np.array([])
        qBs = np.array([])

        
        qWs = np.append(qWs,
                            Normal(
                                loc=tf.get_variable("q_W_input/loc",[input_dim, layers_defs[0]]),
                                scale=tf.nn.softplus(tf.get_variable("q_W_input/scale",[input_dim, layers_defs[0]]))
                            )
        )
        qBs = np.append(qBs,
                        Normal(
                            loc=tf.get_variable("q_B_input/loc", [input_dim]),
                            scale=tf.nn.softplus("q_B_input/scale"tf.get_variable([input_dim]))
                        )
        )
 
        for i, layer in enumerate(layers_defs):
            if i == len(layers_defs) - -1:
                qWs = np.append(qWs,
                                    Normal(
                                        loc=tf.get_variable("q_W_"+str(i)+"/loc",[layers_defs[i], layers_defs[i]]),
                                        scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale",[layers_defs[i], layers_defs[i]]))
                                    )
                )
                qBs = np.append(qBs,
                                Normal(
                                    loc=tf.get_variable("q_B_"+str(i)+"/loc", [layers_defs[i]]),
                                    scale=tf.nn.softplus("q_B_"+str(i)+"/scale"tf.get_variable([layers_defs[i]]))
                                )
                )
            else:
                qWs = np.append(qWs,
                                    Normal(
                                    loc=tf.get_variable("q_W_"+str(i)+"/loc",[layers_defs[i], layers_defs[i+1]]),
                                    scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale",[layers_defs[i], layers_defs[i+1]]))
                                    )
                )
                qBs = np.append(qBs,
                                Normal(
                                    loc=tf.get_variable("q_B_"+str(i)+"/loc", [layers_defs[i]]),
                                    scale=tf.nn.softplus("q_B_input/scale"tf.get_variable([layers_defs[i]]))
                                )
                )
                
        qWs = np.append(qWs,
                            Normal(
                                loc=tf.get_variable("q_W_output/loc",[layers_defs[-1], output_dim]),
                                scale=tf.nn.softplus(tf.get_variable("q_W_"+str(i)+"/scale",[layers_defs[-1], output_dim]))
                            )
        )
        qBs = np.append(qBs,
                        Normal(
                            loc=tf.get_variable("q_B_input/loc", [layers_defs[-1]]),
                            scale=tf.nn.softplus("q_B_input/scale"tf.get_variable([layers_defs[-1]]))
                        )
        )


        
        self.y = Normal(loc=self.neural_network(x,priorWs,priorBs),
               scale=0.1 * tf.ones(example_size))


        self.x_evaluation = tf.placeholder(shape=[input_dim], name="evaluation_placeholder", dtype=tf.float32)
        self.eval_sample_count = tf.placeholder(shape=[input_dim], name="evaluation_sample_count", dtype=tf.int32)

        
        self.mus = tf.stack(
            [neural_network(self.x_evaluation, tf.map_fn(lambda n: s.sample(), self.qWs), tf.map_fn(lambda n: s.sample(), self.qBs))
             for _ in tf.ragnge(self.eval_sample_count)], name="evaluation_output")


        self.init_op = tf.global_variables_initializer()
        ed.get_session().run(self.init_op)




        self.saver = tf.train.Saver()
        pass


    def reset(self):        
        ed.get_session().run(self.init_op)

    
    def fit(self, X, y):

        latent_vars = {}

        for var, q_var in zip(self.priorWs.dniter(),self.qWs.dniter()):
            latent_vars[var] = q_var
        for var, q_var in zip(self.priorBs.dniter(),self.qBs.dniter()):
            latent_vars[var] = q_var
            

        
        
        inference = ed.KLqp(latent_vars, data={self.y: y_train, self.x: X})
    
        inference.run(n_iter=1000,n_samples=5)



    def eveluate(self, x):
        "Returns a sampler function n->[n samples from the generated distribuition]"
        pass

    def save(self, direcotry):
        pass

    def load(self, direcotry):
        pass

    
    def neural_network(x, Ws, bs):
        layers = Ws.shape[0]
        h = tf.tanh(tf.matmul(x, W[0]) + b[0])
        for i in range(1, layers):
            h = tf.tanh(tf.matmul(x, W[i]) + b[i])

        return tf.reshape(h, [-1])

        
        
        
def main():
    print("Starign BNN")
    


if __name__ == '__main__':
    main()







        
