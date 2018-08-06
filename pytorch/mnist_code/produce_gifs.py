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


grid = [(hu, bs) for hu in ([400],[400,400]) for bs in (1,10,100)]

for i, (hidden_sizes, bs) in enumerate(grid):
    
    ################
    ## Aniomation ##
    ################
    
    # Folder for storing results
    results_folder = "./results/"
    
    # Folder containing data
    data_folder = "./../vadam/data"
    
    # Data set
    data_set = "mnist"
    
    # Model parameters
    model_params = {'hidden_sizes': None,
                    'act_func': "relu",
                    'prior_prec': None}
    
    # Training parameters
    train_params = {'num_epochs': None,
                    'batch_size': 100,
                    'train_mc_samples': 10,
                    'eval_mc_samples': 10,
                    'seed': 123}
    
    # Optimizer parameters
    optim_params = {'learning_rate': 0.001,
                    'betas': (0.9,0.999),
                    'prec_init': None}
    optim_params_vogn = {'learning_rate': 0.001,
                         'beta': 0.999,
                         'prec_init': None}
    
    # Evaluations per epoch
    evals_per_epoch = None
    
    animated_gif = AnimatedGif(figsize=figsize)
    plot_every = 10
    
    for prec in (1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2):
        
        model_params['hidden_sizes'] = hidden_sizes
        model_params['prior_prec'] = prec
        optim_params['prec_init'] = prec
        optim_params_vogn['prec_init'] = prec
        train_params['batch_size'] = bs
        if bs==1:
            train_params['num_epochs'] = 2
            evals_per_epoch = 600
        elif bs==10:
            train_params['num_epochs'] = 20
            evals_per_epoch = 60
        elif bs==100:
            train_params['num_epochs'] = 200
            evals_per_epoch = 6
        
        metrics = load_metric_history(experiment_name = "bbb_mlp_class",
                                      data_set = data_set,
                                      model_params = model_params,
                                      train_params = train_params,
                                      optim_params = optim_params,
                                      results_folder = results_folder)
        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        epoch = (idx+1) * train_params['num_epochs'] / num_evals
        met = np.array(metrics['test_pred_logloss'])
        plot_bbb, = plt.plot(epoch, met[idx]/np.log(10), color = 'k', linestyle = '-', linewidth=2)
        
        
        metrics = load_metric_history(experiment_name = "vadam_mlp_class",
                                      data_set = data_set,
                                      model_params = model_params,
                                      train_params = train_params,
                                      optim_params = optim_params,
                                      results_folder = results_folder)
        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        epoch = (idx+1) * train_params['num_epochs'] / num_evals
        met = np.array(metrics['test_pred_logloss'])
        plot_vadam, = plt.plot(epoch, met[idx]/np.log(10), color = 'r', linestyle = '-', linewidth=2)
        plot_title = plt.text(train_params['num_epochs']/2, 2.02, "Precision: " + str(prec), horizontalalignment='center', verticalalignment='bottom', fontdict=dict(fontsize=title_size))
        
        animated_gif.add([plot_bbb, plot_vadam, plot_title])
    
    plt.rc('text', usetex = True)
    plt.xlim([0,train_params['num_epochs']])
    plt.ylim([0,2])
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("Epoch", fontdict=dict(fontsize=label_size))
    plt.ylabel(r'Test $\log_{10}$loss', fontdict=dict(fontsize=label_size))
    plt.legend(["BBVI", "Vadam"], fontsize=legend_size, loc='upper center')
    plt.grid(True,which="both",color='0.75')
    plt.tight_layout()
    animated_gif.save('animations/layer' + str(len(hidden_sizes)) + '_batchsize'+str(bs)+'.gif')