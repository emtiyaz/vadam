# Size of figure
#figsize = (6, 4)
figsize = (4,5.8)

# Font sizes
title_size = 24
legend_size = 22
label_size = 22
tick_size = 22

##################
## Load Imports ##
##################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.ioff()
#plt.ion()

from experiments import load_metric_history

def plot_init(fig_size, xlabel = "Epoch", ylabel = "\log_base", log_base = None, title = ""):
    fig = plt.figure(figsize = fig_size)
    plt.rc('text', usetex = True)
    plt.grid(True,which="both",color='0.75')
    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel == "\log_base":
        if log_base == None:
            plt.ylabel(r'$\log$loss')
        else:
            plt.ylabel(r'$\log_{' + str(log_base) + '}$loss')
    else:
        plt.ylabel(ylabel)
    return fig

plot_every = 10
def plot_metrics(axis, keys, colors, linestyles, results_folder, experiment_name, data_set, model_params, train_params, optim_params, log_base=np.exp(1), *args, **kwargs):
    metrics = load_metric_history(experiment_name = experiment_name,
                                  data_set = data_set,
                                  model_params = model_params,
                                  train_params = train_params,
                                  optim_params = optim_params,
                                  results_folder = results_folder)
    num_evals = len(metrics[keys[0]])
    idx = np.arange(start = 0, stop = num_evals, step = plot_every)
    for i, key in enumerate(keys):
        met = np.array(metrics[key])
        axis.plot(idx+1, met[idx]/np.log(log_base), color = colors[i], linestyle = linestyles[i], *args, **kwargs)
        

np.random.seed(111)

######################
## Interactive Plot ##
######################

def plot_results(batch_size, train_mc_samples, legend=True):
    
    num_epochs = 16 * batch_size
    
    ####################
    ## Set parameters ##
    ####################
    
    # Dump pdfs to folder
    pdf_folder = "./plots/"
    
    # Folder for storing results
    results_folder = "./results/"

    # Data set
    data_set = "australian_presplit"
    log_base = 2

    # Model parameters
    model_params = {'hidden_sizes': [64],
                    'act_func': 'relu',
                    'prior_prec': 1.0}

    # Training parameters
    train_params = {'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'train_mc_samples': train_mc_samples,
                    'eval_mc_samples': 10,
                    'seed': 123}

    # Optimization parameters
    optim_params = {'learning_rate': 0.001,
                    'betas': (0.9, 0.999),
                    'prec_init': 1.0}
    optim_params_ggn = {'learning_rate': 0.001,
                        'beta': 0.999,
                        'prec_init': 1.0}

    ##################
    ## Plot logloss ##
    ##################

    fig = plot_init(fig_size = figsize, log_base = log_base)
    plot_metrics(plt, 
                 keys = ['test_pred_logloss'], 
                 log_base = log_base, 
                 colors = ['k'], 
                 linestyles = ['--'], 
                 results_folder = results_folder, 
                 experiment_name = "bbb_mlp_binclass", 
                 data_set = data_set, 
                 model_params = model_params,
                 train_params = train_params, 
                 optim_params = optim_params, 
                 linewidth = 2.0)
    plot_metrics(plt, 
                 keys = ['test_pred_logloss'], 
                 log_base = log_base, 
                 colors = ['r'], 
                 linestyles = ['-'], 
                 results_folder = results_folder, 
                 experiment_name = "vadam_mlp_binclass", 
                 data_set = data_set, 
                 model_params = model_params,
                 train_params = train_params, 
                 optim_params = optim_params, 
                 linewidth = 2.0)
    plot_metrics(plt, 
                 keys = ['test_pred_logloss'], 
                 log_base = log_base, 
                 colors = ['g'], 
                 linestyles = ['-'], 
                 results_folder = results_folder, 
                 experiment_name = "vogn_mlp_binclass", 
                 data_set = data_set, 
                 model_params = model_params,
                 train_params = train_params, 
                 optim_params = optim_params_ggn, 
                 linewidth = 2.0)
    
    if legend:
        plt.legend(["BBVI", "Vadam", "VOGN"], fontsize=legend_size)
    ax = fig.axes[0]
    ax.set_title("M = " + str(batch_size) + ", S = " + str(train_mc_samples), fontsize=title_size)
    ax.set_xlabel('Iteration', fontdict = {'fontsize': label_size})
    ax.set_ylabel(r'Test $\log_{2}$loss', fontdict = {'fontsize': label_size})
    ax.tick_params(labelsize=tick_size)
    plt.ylim(0.5, 2.0)
    plt.xlim(0, 5000)
    plt.yticks([0.5, 1.0, 1.5, 2.0],fontsize=tick_size)
    plt.tight_layout()
    if legend:
        plt.savefig(pdf_folder + "plot_bs" + str(batch_size) + "_mc" + str(train_mc_samples) + "_legend.pdf", bbox_inches='tight')
    else:
        plt.savefig(pdf_folder + "plot_bs" + str(batch_size) + "_mc" + str(train_mc_samples) + ".pdf", bbox_inches='tight')



plot_results(batch_size = 1, 
             train_mc_samples = 1, 
             legend = False)
plot_results(batch_size = 1, 
             train_mc_samples = 16, 
             legend = False)
plot_results(batch_size = 128, 
             train_mc_samples = 16, 
             legend = True)
