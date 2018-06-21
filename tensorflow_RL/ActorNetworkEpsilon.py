"""
This script implements Actor Network with stochastic weights for DDPG
"""

import tensorflow as tf
import numpy as np
import tflearn

class ActorNetworkEpsilon(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, method_name, 
                 learning_rate, actor_lr_sigma, tau, prec_init, gamma_2, prior=1e-8, gamma_1=0.9, min_prec=10000):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma_2 = gamma_2
        self.prec_init = prec_init

        self.prior = prior
        self.gamma_1 = gamma_1
        self.min_prec = min_prec
        self.is_vadam = False

        if method_name.lower() == "vadam": 
            self.is_vadam = True
            self.gamma_2_power_t_value = 1.0
            self.gamma_1_power_t_value = 1.0

        """" Actor Network, with noise on parameter """
        scope1 = 'actor'        
        self.inputs, self.out, self.scaled_out, self.eps_w1, self.eps_w2, self.eps_w3, self.eps_b1, self.eps_b2, self.eps_b3 = \
            self.create_noise_network(scope=scope1, action_bound=action_bound)
        self.mean_params = tf.get_collection(scope1)
        self.variance_params = tf.get_collection(scope1+'_prec')

        """ Target Network, no noise """
        scope2 = 'target_actor'
        self.target_inputs, self.target_out, self.target_scaled_out = \
            self.create_nonoise_network(scope=scope2, action_bound=action_bound)
        self.target_network_params = tf.get_collection(scope2)

        """ Test network, no noise """
        scope3 = 'test'
        self.test_inputs, self.test_out, self.test_scaled_out = \
            self.create_nonoise_network(scope=scope3, action_bound=action_bound)
        self.test_network_params = tf.get_collection(scope3)

        """ Op. for soft updating target network """
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.mean_params[i], tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - tau))
                for i in range(len(self.target_network_params))]

        """ Op. for hard updating test network """
        self.update_test_network_params = \
            [self.test_network_params[i].assign(self.mean_params[i])
                for i in range(len(self.test_network_params))]

        """ This gradient will be provided by the critic network """
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        self.sess.run(tf.global_variables_initializer())

        self.num_trainable_vars = len(self.mean_params) + len(self.variance_params) + \
                                  len(self.target_network_params) + len(self.test_network_params)

        """ gradietn computation Op. """
        self.mean_grads = tf.gradients(self.scaled_out, self.mean_params, -self.action_gradient)
        self.variance_grads = tf.gradients(self.scaled_out, self.variance_params, -self.action_gradient)

        self.var_holder = [ tf.placeholder(dtype=tf.float32,shape= self.variance_params[i].shape)
                            for i in range(len(self.variance_params)) ]

        self.mean_holder = [tf.placeholder(dtype=tf.float32,shape= self.mean_params[i].shape)
                            for i in range(len(self.mean_params))]

        if method_name.lower() == 'noise+sgd':
            self.optimize_variance = tf.train.GradientDescentOptimizer(learning_rate=actor_lr_sigma).apply_gradients(zip(self.variance_grads, self.variance_params))
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(self.mean_grads, self.mean_params))
        elif method_name.lower() == "noise+adam":
            self.optimize_variance = tf.train.AdamOptimizer(learning_rate=actor_lr_sigma).apply_gradients(zip(self.variance_grads, self.variance_params))
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(self.mean_grads, self.mean_params))

        elif method_name.lower() == 'vadagrad':
            self.optimize_variance = self.vadagrad_update_prec(grad_prec=self.var_holder, prec=self.variance_params)
            self.optimize = self.vadagrad_update_mean(grad=self.mean_holder, var=self.mean_params, prec=self.variance_params)

        elif method_name.lower() == "vadam":                
            self.mean_momentum = [tf.get_variable("mean_momentum%d" % i, dtype=tf.float32, shape=self.mean_params[i].shape, initializer=tf.zeros_initializer, trainable=False) for i in range(len(self.mean_params))]
            self.gamma_2_power_t = tf.Variable(1.0, dtype=tf.float32, trainable=False)
            self.gamma_1_power_t = tf.Variable(1.0, dtype=tf.float32, trainable=False)

            self.optimize_variance = self.vadam_update_prec(prec=self.variance_params, grad_prec=self.var_holder)
            self.optimize_mean_momentum = self.vadam_update_mean_momentum(mean_momentum=self.mean_momentum, grad_mean=self.mean_holder, mean=self.mean_params)
            self.optimize_mean = self.vadam_update_mean(mean=self.mean_params, mean_momentum=self.mean_momentum, prec=self.variance_params)
        else:
            raise NotImplementedError

        """ Op for computing statistics during learning """
        self.min_variance = [tf.reduce_min(self.variance_params[check_i]) for check_i in range(len(self.variance_params))]
        self.max_variance = [tf.reduce_max(self.variance_params[check_i]) for check_i in range(len(self.variance_params))]

    """ Vadam """
    # Update the mean of Gaussian using the first-order moment (mean_momentum) and the second-order moment (prec).
    def vadam_update_mean(self, mean, mean_momentum, prec):
        inverse_prec_params = [1/(tf.sqrt(a) + self.prior) for a in prec]
        natural_grads = [tf.multiply(a, b) for a, b in zip(inverse_prec_params, mean_momentum)]

        update_mean = [mean[i].assign(mean[i] - (self.learning_rate * tf.sqrt( 1 - self.gamma_2_power_t) / (1 - self.gamma_1_power_t)) * natural_grads[i]) for i in range(len(mean))]
        return update_mean   

    # update first-order moment by mean_momentum <-- gamma_1*mean_momentum + (1-gamma_1)*( grad_mean + lambda * mean )
    def vadam_update_mean_momentum(self, mean_momentum, grad_mean, mean):
        update_mean_momentum = [ mean_momentum[i].assign( self.gamma_1 * mean_momentum[i] + (1-self.gamma_1) * (grad_mean[i] + self.prior*mean[i]) ) for i in range(len(mean_momentum))]
        return update_mean_momentum     

    # update second-order moment by prec <-- gamma_2*prec + (1-gamma_2)*( grad_prec )
    def vadam_update_prec(self, prec, grad_prec):
        update_prec = [ prec[i].assign( self.gamma_2 * prec[i] + (1-self.gamma_2) * grad_prec[i]) for i in range(len(prec))]
        return update_prec 
      
    def train_vadam(self, variance_grad, mean_grad):

        self.gamma_2_power_t_value = self.gamma_2_power_t_value * self.gamma_2
        self.gamma_1_power_t_value = self.gamma_1_power_t_value * self.gamma_1

        self.gamma_2_power_t.load(self.gamma_2_power_t_value, self.sess)
        self.gamma_1_power_t.load(self.gamma_1_power_t_value, self.sess)

        self.sess.run(self.optimize_variance, feed_dict={
            i:d for i, d in zip(self.var_holder, variance_grad)
        })

        self.sess.run(self.optimize_mean_momentum, feed_dict={
            i:d for i, d in zip(self.mean_holder, mean_grad)
        })

        self.sess.run(self.optimize_mean)
    
    """ Vadagrad """
    def vadagrad_update_mean(self, grad, var, prec):
        inverse_var_params = [1/tf.sqrt(a) for a in prec]
        natural_grads = [tf.multiply(a, b) for a, b in zip(inverse_var_params, grad)]
        update_mean = [var[i].assign(var[i] - self.learning_rate * natural_grads[i]) for i in range(len(var))]
        return update_mean   

    def vadagrad_update_prec(self, grad_prec, prec):
        update_prec = [ prec[i].assign(prec[i] + (1 - self.gamma_2) * grad_prec[i]) for i in range(len(prec))]
        return update_prec 
        
    def train_vadagrad(self, variance_grad, mean_grad):
        self.sess.run(self.optimize_variance, feed_dict={
            i:d for i, d in zip(self.var_holder, variance_grad)
        })
        self.sess.run(self.optimize, feed_dict = {
            i:d for i, d in zip(self.mean_holder, mean_grad)
        })

    """ noise+sgd and noise+adam """
    def train_noise(self, inputs, a_gradient, eps_list):
        self.sess.run(self.optimize_variance, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.eps_w1:eps_list[0],
            self.eps_w2:eps_list[1],
            self.eps_w3:eps_list[2],
            self.eps_b1:eps_list[3],
            self.eps_b2:eps_list[4],
            self.eps_b3:eps_list[5],
        })

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.eps_w1:eps_list[0],
            self.eps_w2:eps_list[1],
            self.eps_w3:eps_list[2],
            self.eps_b1:eps_list[3],
            self.eps_b2:eps_list[4],
            self.eps_b3:eps_list[5],
        })

    """ comput action from policy network with sampled weights """
    def predict(self, inputs, eps_list):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.eps_w1:eps_list[0],
            self.eps_w2:eps_list[1],
            self.eps_w3:eps_list[2],
            self.eps_b1:eps_list[3],
            self.eps_b2:eps_list[4],
            self.eps_b3:eps_list[5],
        })

    """ compute action from target policy network """
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    """ compute action from test policy network """
    def predict_test(self, inputs):
        return self.sess.run(self.test_scaled_out, feed_dict={
            self.test_inputs: inputs
        })

    """ update target policy network by moving average """
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    """ update test policy network by hard copy """
    def update_test_network(self):
        self.sess.run(self.update_test_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    """ get minimum and maximum value of the precision vector s"""
    def get_min_max_sum(self):
        #self.sess.run(self.reducesum)
        a = self.sess.run(self.min_variance)
        b = self.sess.run(self.max_variance)

        return np.min(np.array(a)), np.max(np.array(b))

    """ Compute averaged gradient w.r.t. weight parameter. Used by VadaGrad and Vadam to compute squared of averaged gradients """
    def get_grad(self, inputs, a_gradient, eps_list):
        mean_grad = self.sess.run(self.mean_grads, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.eps_w1:eps_list[0],
            self.eps_w2:eps_list[1],
            self.eps_w3:eps_list[2],
            self.eps_b1:eps_list[3],
            self.eps_b2:eps_list[4],
            self.eps_b3:eps_list[5],
        })

        return mean_grad
        
    """ create policy network with noise weight """
    def create_noise_network(self, scope, action_bound=1):
        state_inputs = tf.placeholder(dtype=tf.float32,shape=(None, self.s_dim),name='input')
        eps_w_list = []
        eps_b_list = []
        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        prec_initializer = tf.constant_initializer(value=self.prec_init)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        b_initializer = tf.constant_initializer(value=0.1)
        n_layer1 = 400
        n_layer2 = 300
        collection = [scope, tf.GraphKeys.GLOBAL_VARIABLES]


        eps_w1 = tf.placeholder(shape=(self.s_dim,n_layer1), dtype=tf.float32)
        eps_w2 = tf.placeholder(shape=(n_layer1,n_layer2), dtype=tf.float32)
        eps_w3 = tf.placeholder(shape=(n_layer2,self.a_dim), dtype=tf.float32)

        eps_b1 = tf.placeholder(shape=(1, n_layer1), dtype=tf.float32)
        eps_b2 = tf.placeholder(shape=(1, n_layer2), dtype=tf.float32)
        eps_b3 = tf.placeholder(shape=(1, self.a_dim), dtype=tf.float32)

        eps_list = [eps_w1, eps_w2, eps_w3, eps_b1, eps_b2, eps_b3]


        def build_layer(layer_scope,dim_1,dim_2,input,eps_w, eps_b, collections, output_layer, w_initializer = w_initializer):
            with tf.variable_scope(layer_scope):
                if output_layer:
                    w_initializer = w_init
                w = tf.get_variable(name='w',shape=(dim_1,dim_2),dtype=tf.float32,
                                 initializer=w_initializer,collections=collections)
                b = tf.get_variable(name='b',shape=(1, dim_2),dtype=tf.float32,
                                 initializer= b_initializer, collections=collections)

                prec_w = tf.get_variable(name='prec_w',shape=(dim_1,dim_2),dtype=tf.float32,
                             initializer=prec_initializer,collections=[scope+'_prec', tf.GraphKeys.GLOBAL_VARIABLES])

                prec_b = tf.get_variable(name='prec_b',shape=(1,dim_2),dtype=tf.float32,
                             initializer=prec_initializer,collections=[scope+'_prec', tf.GraphKeys.GLOBAL_VARIABLES])

                if not self.is_vadam:
                    noisy_w = w + eps_w * (1/tf.sqrt(prec_w))
                    noisy_b = b + eps_b * (1/tf.sqrt(prec_b))
                else:
                    noisy_w = w + eps_w * (1/tf.sqrt(prec_w + self.prior + self.min_prec))
                    noisy_b = b + eps_b * (1/tf.sqrt(prec_b + self.prior + self.min_prec))


                if output_layer:
                    layer_output = tf.nn.tanh(tf.matmul(input, noisy_w) + noisy_b)
                else:
                    layer_output = tf.nn.relu(tf.matmul(input, noisy_w) + noisy_b)

                return layer_output

        with tf.variable_scope(scope):

            layer_1_output = build_layer(layer_scope='layer_1',dim_1=self.s_dim,dim_2=n_layer1,input=state_inputs,
                                         eps_w=eps_list[0], eps_b=eps_list[3], collections=collection,output_layer=False)
            layer_2_output = build_layer(layer_scope='layer_2',dim_1=n_layer1,dim_2=n_layer2,input=layer_1_output,
                                         eps_w=eps_list[1], eps_b=eps_list[4],collections=collection,output_layer=False)
            output= build_layer(layer_scope='layer_3',dim_1=n_layer2,dim_2=self.a_dim,input=layer_2_output,
                                         eps_w=eps_list[2], eps_b=eps_list[5],collections=collection,output_layer=True)

            scaled_out = tf.multiply(output, action_bound)

        return state_inputs, output, scaled_out, eps_w1, eps_w2, eps_w3, eps_b1, eps_b2, eps_b3

    """ create policy network with deterministic weight """
    def create_nonoise_network(self, scope, action_bound=1):

        state_inputs = tf.placeholder(dtype=tf.float32,shape=(None, self.s_dim),name='input')
        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        #prec_initializer = tf.constant_initializer(value=self.prec_init) # not used
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        b_initializer = tf.constant_initializer(value=0.1)
        n_layer1 = 400
        n_layer2 = 300
        collection = [scope, tf.GraphKeys.GLOBAL_VARIABLES]

        def build_layer(layer_scope,dim_1,dim_2,input,collections, output_layer, w_initializer = w_initializer):
            with tf.variable_scope(layer_scope):
                if output_layer:
                    w_initializer = w_init

                w = tf.get_variable(name='w',shape=(dim_1,dim_2),dtype=tf.float32,
                                 initializer=w_initializer,collections=collections)
                b = tf.get_variable(name='b',shape=(1, dim_2),dtype=tf.float32,
                                 initializer= b_initializer, collections=collections)

                if output_layer:
                    layer_output = tf.nn.tanh(tf.matmul(input, w) + b)
                else:
                    layer_output = tf.nn.relu(tf.matmul(input, w) + b)

                return layer_output

        with tf.variable_scope(scope):
            layer_1_output = build_layer(layer_scope='layer_1',dim_1=self.s_dim,dim_2=n_layer1,input=state_inputs,
                                         collections=collection,output_layer=False)
            layer_2_output = build_layer(layer_scope='layer_2',dim_1=n_layer1,dim_2=n_layer2,input=layer_1_output,
                                         collections=collection,output_layer=False)
            output = build_layer(layer_scope='layer_3',dim_1=n_layer2,dim_2=self.a_dim,input=layer_2_output,
                                         collections=collection,output_layer=True)

            scaled_out = tf.multiply(output, action_bound)

        return state_inputs, output, scaled_out
