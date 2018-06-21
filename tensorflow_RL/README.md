# Tensorflow implementation of Parameter-based Exploration DDPG via VadaGrad and Vadam

This directory contains tensorflow implementation of the deep RL experiment in the paper.

## Requirements
The experiment is performed on Python 3.5.3 with these python packages:
* [tensorflow](https://github.com/tensorflow/tensorflow) == 1.2.1
* [numpy](http://www.numpy.org/) == 1.12.1
* [tflearn](http://tflearn.org/) == 0.3.2
* [gym](https://github.com/openai/gym) == 0.9.1
* [mujoco-py](https://github.com/openai/mujoco-py) == 0.5.7

## Scripts explanation:
There are 6 python scripts:
* DDPG.py - This is the main script to run experiment for all methods.
* ActorNetworkEpsilon.py - This script implements actor network with stochastic weights (see the remark below). This is used by VadaGrad, Vadam, SGD-Explore, and Adam-Explore.
* ActorNetworkNoNoise.py - This script implements actor network with deterministic weights. This is the actor network for usual DDPG, and is used by SGD-Plain and Adam-Plain.
* CriticNetwork.py - This script implements critic network. This is the critic network for usual DDPG, and is used by all methods.
* Replay_buffer.py - This implements the usual first-in-first-out replay buffer.
* plot_halfcheetah.py - This is for plotting the result. It loads .npy test performance files and plot averaged performances.

### How to train an agent:
Train an agent is done via running DDPG.py.
For example, to train with Vadam for HalfCheetah-v1, run
```
python DDPG.py --method vadam --env_name HalfCheetah-v1
```
You can change the method value to the followings (case insensitive): 'vadam', 'vadagrad', 'noise+sgd', 'noise+adam', 'sgd', and 'adam'.
Note that the name is slightly different in the paper: 'noise+sgd' is SGD-Explore, 'sgd' is SGD-Plain, 'noise+adam' is Adam-Explore, and 'adam' is Adam-Plain.

By default, the agent is trained for 3000 episodes with the default hyper-parameters reported in the paper.
Once every 10 training episodes, a test performance is evaluated by running 20 episodes with a deterministic actor network (called test_network in the code).
The test performance is then saved in the result directory as a .npy file. 
A trained policy is also saved in the result directory.
The results in the paper are obtained using random seeds 0 to 4.

### How to plot the result:
The result reported in the paper is provided in 'result/HalfCheetah-v1' directory.
Run
```
python plot_halfcheetah.py
```
The script loads test returns in .npy files, and plots the mean and stadard error over 5 random seeds.
![](tensorflow_RL/result/HalfCheetah-v1.pdf "HalfCheetah")

## Remarks
### Reproducability of the result:
The result is reproducible up to some extent due to non-deterministicities of tensorflow-gpu in older versions of tensorflow.
Even with the same random seeds and same GPU machine, we could not run the same code to have an identical result for differnt runs.
Though, similar learning curves should appear. ('result/HalfCheetah-v1_2nd' directory contains experimental results for different runs with the same random seeds.)
![](tensorflow_RL/result/HalfCheetah-v1_2nd.pdf "HalfCheetah_2nd")
The non-deterministicty is fixed in newer versions of tensorflow (https://github.com/tensorflow/tensorflow/issues/12731) but we have not tried the new version yet.
We are planning to re-implement this experiment in pytorch in order to use the same implementation as in our Bayseian neural network experiments.

### Implementing stochastic network weights
We found that it is non-trivial to implement controllable stochastic weights in tensorflow.
It is possible to use tf.random_normal variable when creating computation graphs, but we cannot easily control when the weights are sampled.
To have more control (and more transparency during debugging), we implemented a network that take noise as input (see line 152 in DDPG.py and 241 in ActorNetworkEpsilon.py).
This also allows us to use the same noise for computing action and gradient, and is easily extenable to use more than 1 sampled weights (i.e., more than 1 MC samples from variational distribution).

### Results on other tasks.
Beside HalfCheetah-v1, we did not evaluate other tasks. So it is possible that the default hyper-parameters do not work well on other tasks.

## Acknowledgements:
* Zuozhu Liu (SUTD, Singapore) for the initial implementation.
* The implementation is largely based on a nice DDPG implementation by Patrick Emami:
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html and https://github.com/pemami4911/deep-rl

