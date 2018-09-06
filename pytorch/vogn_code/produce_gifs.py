# Size of figure
figsize = (6, 4)

# Font sizes
title_size = 14
legend_size = 16
label_size = 14
tick_size = 14

##################
## Load Imports ##
##################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as anim
plt.ioff()
#plt.ion()

from experiments import load_metric_history
        
class AnimatedGif:
    def __init__(self, figsize=(12, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.images = []
 
    def add(self, plot_list):
        self.images.append(plot_list)
 
    def save(self, filename, fps=1):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=fps)
        
np.random.seed(111)


grid = [mc for mc in (1, 16)]

for i, (mc) in enumerate(grid):
    
    ################
    ## Aniomation ##
    ################
    
    # Folder for storing results
    results_folder = "./results/"
    
    # Folder containing data
    data_folder = "./../vadam/data"
    
    # Data set
    data_set = "australian_presplit"
    
    # Model parameters
    model_params = {'hidden_sizes': [64],
                    'act_func': "relu",
                    'prior_prec': 1.0}
    
    # Training parameters
    train_params = {'num_epochs': None,
                    'batch_size': None,
                    'train_mc_samples': mc,
                    'eval_mc_samples': 10,
                    'seed': 123}
    
    # Optimizer parameters
    optim_params = {'learning_rate': 0.001,
                    'betas': (0.9,0.999),
                    'prec_init': 1.0}
    optim_params_vogn = {'learning_rate': 0.001,
                         'beta': 0.999,
                         'prec_init': 1.0}
    
    # Evaluations per epoch
    evals_per_epoch = 1000
    
    animated_gif = AnimatedGif(figsize=figsize)
    plot_every = 10
    
    for bs in (1, 4, 16, 128):
        
        train_params['batch_size'] = bs
        train_params['num_epochs'] = 16 * bs
        
        metrics = load_metric_history(experiment_name = "bbb_mlp_binclass",
                                      data_set = data_set,
                                      model_params = model_params,
                                      train_params = train_params,
                                      optim_params = optim_params,
                                      results_folder = results_folder)
        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        met = np.array(metrics['test_pred_logloss'])
        plot_bbb, = plt.plot(idx, met[idx]/np.log(2), color = 'k', linestyle = '-', linewidth=2)
        
        
        metrics = load_metric_history(experiment_name = "vadam_mlp_binclass",
                                      data_set = data_set,
                                      model_params = model_params,
                                      train_params = train_params,
                                      optim_params = optim_params,
                                      results_folder = results_folder)
        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        met = np.array(metrics['test_pred_logloss'])
        plot_vadam, = plt.plot(idx, met[idx]/np.log(2), color = 'r', linestyle = '-', linewidth=2)
        
        
        metrics = load_metric_history(experiment_name = "vogn_mlp_binclass",
                                      data_set = data_set,
                                      model_params = model_params,
                                      train_params = train_params,
                                      optim_params = optim_params_vogn,
                                      results_folder = results_folder)
        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        met = np.array(metrics['test_pred_logloss'])
        plot_vogn, = plt.plot(idx, met[idx]/np.log(2), color = 'g', linestyle = '-', linewidth=2)
        plot_title = plt.text(5000/2, 2.02, "Batch Size: " + str(bs), horizontalalignment='center', verticalalignment='bottom', fontdict=dict(fontsize=title_size))
        
        animated_gif.add([plot_bbb, plot_vadam, plot_vogn, plot_title])
    
    plt.rc('text', usetex = True)
    plt.xlim(0, 5000)
    plt.ylim(0.5, 2.0)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("Iteration", fontdict=dict(fontsize=label_size))
    plt.ylabel(r'Test $\log_{2}$loss', fontdict=dict(fontsize=label_size))
    plt.legend(["BBVI", "Vadam", "VOGN"], fontsize=legend_size, loc='upper right')
    plt.grid(True,which="both",color='0.75')
    plt.tight_layout()
    animated_gif.save('animations/mc' + str(mc) + '.gif')