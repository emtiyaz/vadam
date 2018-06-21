"""
This script implements Actor Network with deterministic weights for DDPG, i.e., original DDPG.
"""

import tensorflow as tf
import numpy as np
import tflearn

class ActorNetworkNoNoise(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, method_name,
                 learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate

        """" Actor Network, no noise """
        self.scope1 = 'actor'
        self.inputs, self.out, self.scaled_out = self.create_nonoise_network(scope=self.scope1, action_bound=action_bound)
        self.mean_params = tf.get_collection(self.scope1)

        """ Target Network, no noise """
        self.scope2 = 'target_actor'
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_nonoise_network(scope=self.scope2, action_bound=action_bound)
        self.target_network_params = tf.get_collection(self.scope2)

        """ Test network, no noise """
        self.scope3 = 'test'
        self.test_inputs, self.test_out, self.test_scaled_out = self.create_nonoise_network(scope=self.scope3, action_bound=action_bound)
        self.test_network_params = tf.get_collection(self.scope3)


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

        self.num_trainable_vars = len(self.mean_params) + \
                                  len(self.target_network_params) + len(self.test_network_params)

        """ gradietn computation Op. """
        self.mean_grads = tf.gradients(self.scaled_out, self.mean_params, -self.action_gradient)

        if method_name.lower() == "sgd":
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))
        elif method_name.lower() == "adam":
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))

    def create_nonoise_network(self, scope, action_bound=1):

        state_inputs = tf.placeholder(dtype=tf.float32,shape=(None, self.s_dim),name='input')
        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        #prec_initializer = tf.constant_initializer(value=self.noise) # not used
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

    def train(self, inputs, a_gradient):
    
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def predict_test(self, inputs):
        return self.sess.run(self.test_scaled_out, feed_dict={
            self.test_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def update_test_network(self):
        self.sess.run(self.update_test_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
