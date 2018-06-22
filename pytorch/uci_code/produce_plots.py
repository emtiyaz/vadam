import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

##################
## Define funcs ##
##################

def folder_name(experiment_name, param_bounds, bo_params, data_params, model_params, train_params, optim_params, results_folder="./results"):
    pp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(param_bounds.items()))[:-1]
    bp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(bo_params.items()))[:-1]
    dp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(data_params.items()))[:-1]
    mp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(model_params.items()))[:-1]
    tp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(train_params.items()))[:-1]
    op = ''.join('{}:{}|'.format(key, val) for key, val in sorted(optim_params.items()))[:-1]
    return os.path.join(results_folder, experiment_name, pp, bp, dp, mp, tp, op)
    


#####################
## Define datasets ##
#####################

ds = ("boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "wine", "yacht")
plt.ioff()

for data_sets in ds:
    
    # Dump pdfs to folder
    pdf_folder = "./plots/"
    
    # Folder for storing results
    results_folder = "./results/"
    
    # Data set
    data_params = {'data_set': None,
                   'n_splits': 5,
                   'seed': 123}
    
    # Model parameters
    model_params = {'hidden_sizes': [50],
                    'act_func': 'relu',
                    'prior_prec': None,
                    'noise_prec': None}
    
    # BO parameters
    bo_params = {'acq': 'ei',
                 'init_points': 5, 
                 'n_iter': 25}
    
    if data_sets in ("naval", "powerplant"):
        param_bounds = {'log_noise_prec': (1, 5), 
                        'log_prior_prec': (-3, 4)}
    else:
        param_bounds = {'log_noise_prec': (0, 5), 
                        'log_prior_prec': (-4, 4)}
        
    if data_sets in ("kin8nm", "naval", "powerplant", "wine"):
        bbs = 128
        bmc = 10
        
        vbs = 128
        vmc = 5
    else:
        bbs = 32
        bmc = 20
        
        vbs = 32
        vmc = 10
    
    
    
    #######################
    ## Load BBVI results ##
    #######################
        
    experiment_name = "bayesopt_bbb"
    
    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': bbs,
                    'train_mc_samples': bmc,
                    'eval_mc_samples': 100,
                    'seed': 123}
    
    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'betas': (0.9,0.99),
                    'prec_init': 10.0}
        
    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    
    for i, (data_set) in enumerate(grid_marginalize):
        
        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name, 
                             param_bounds = param_bounds, 
                             bo_params = bo_params, 
                             data_params = data_params, 
                             model_params = model_params, 
                             train_params = train_params, 
                             optim_params = optim_params)
        
        pkl_file = open(os.path.join(folder, 'metric_history.pkl'), 'rb')
        metric_history = pickle.load(pkl_file)
        pkl_file.close()
        
        if i==0:
            num_evals = len(metric_history['test_pred_logloss'])
            bll = np.zeros([len(grid_marginalize), num_evals])
            brmse = np.zeros([len(grid_marginalize), num_evals])
        
        bll[i] = metric_history['test_pred_logloss']
        brmse[i] = metric_history['test_pred_rmse']
        
        
    bepoch = (1 + np.arange(num_evals)) * train_params['num_epochs'] / (num_evals)
    
    
    
    ########################
    ## Load Vadam results ##
    ########################
        
    experiment_name = "bayesopt_vadam"
    
    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': vbs,
                    'train_mc_samples': vmc,
                    'eval_mc_samples': 100,
                    'seed': 123}
    
    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'betas': (0.9,0.99),
                    'prec_init': 10.0}
        
    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    
    for i, (data_set) in enumerate(grid_marginalize):
        
        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name, 
                             param_bounds = param_bounds, 
                             bo_params = bo_params, 
                             data_params = data_params, 
                             model_params = model_params, 
                             train_params = train_params, 
                             optim_params = optim_params)
        
        pkl_file = open(os.path.join(folder, 'metric_history.pkl'), 'rb')
        metric_history = pickle.load(pkl_file)
        pkl_file.close()
        
        if i==0:
            num_evals = len(metric_history['test_pred_logloss'])
            vll = np.zeros([len(grid_marginalize), num_evals])
            vrmse = np.zeros([len(grid_marginalize), num_evals])
        
        vll[i] = metric_history['test_pred_logloss']
        vrmse[i] = metric_history['test_pred_rmse']
        
        
    vepoch = (1 + np.arange(num_evals)) * train_params['num_epochs'] / (num_evals)
    
    
    
    ########################
    ## Load Vprop results ##
    ########################
        
    experiment_name = "final_vprop"
    
    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': vbs,
                    'train_mc_samples': vmc,
                    'eval_mc_samples': 100,
                    'seed': 123}
    
    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'beta': 0.99,
                    'prec_init': 10.0}
        
    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    
    for i, (data_set) in enumerate(grid_marginalize):
        
        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name, 
                             param_bounds = param_bounds, 
                             bo_params = bo_params, 
                             data_params = data_params, 
                             model_params = model_params, 
                             train_params = train_params, 
                             optim_params = optim_params)
        
        pkl_file = open(os.path.join(folder, 'metric_history.pkl'), 'rb')
        metric_history = pickle.load(pkl_file)
        pkl_file.close()
        
        if i==0:
            num_evals = len(metric_history['test_pred_logloss'])
            pll = np.zeros([len(grid_marginalize), num_evals])
            prmse = np.zeros([len(grid_marginalize), num_evals])
        
        pll[i] = metric_history['test_pred_logloss']
        prmse[i] = metric_history['test_pred_rmse']
        
    
    pepoch = (1 + np.arange(num_evals)) * train_params['num_epochs'] / (num_evals)
    

    
    ############################
    ## Prepare means and stds ##
    ############################
    
    bllm = -np.mean(bll, axis=0)
    blls = np.std(bll, axis=0)/np.sqrt(20)
    brmsem = np.mean(brmse, axis=0)
    brmses = np.std(brmse, axis=0)/np.sqrt(20)
    
    vllm = -np.mean(vll, axis=0)
    vlls = np.std(vll, axis=0)/np.sqrt(20)
    vrmsem = np.mean(vrmse, axis=0)
    vrmses = np.std(vrmse, axis=0)/np.sqrt(20)
    
    pllm = -np.mean(pll, axis=0)
    plls = np.std(pll, axis=0)/np.sqrt(20)
    prmsem = np.mean(prmse, axis=0)
    prmses = np.std(prmse, axis=0)/np.sqrt(20)
    
    
    idx = np.round(np.linspace(start=0, stop=len(bllm)-1, num=100)).astype(int)
    bepoch = bepoch[idx]
    bllm = bllm[idx]
    blls = blls[idx]
    brmsem = brmsem[idx]
    brmses = brmses[idx]
    idx = np.round(np.linspace(start=0, stop=len(vllm)-1, num=100)).astype(int)
    vepoch = vepoch[idx]
    vllm = vllm[idx]
    vlls = vlls[idx]
    vrmsem = vrmsem[idx]
    vrmses = vrmses[idx]
    idx = np.round(np.linspace(start=0, stop=len(pllm)-1, num=100)).astype(int)
    pepoch = pepoch[idx]
    pllm = pllm[idx]
    plls = plls[idx]
    prmsem = prmsem[idx]
    prmses = prmses[idx]
    
    
    
    ##################
    ## Plot results ##
    ##################
    
    fig_size = (4,3)
    set_y_axis = True
    
    lw = 2
    
    grid_color = '0.7'
    grid_lw = 0.2
    
    title_size = 16
    label_size = 16
    tick_size = 14
    x_ticks = np.arange(0,41,step=10)
    
    if set_y_axis == True:
        ylim_ll = [2.5,5]
        yticks_ll = np.arange(ylim_ll[0],ylim_ll[1],step=0.5)
        
        if data_sets=="boston":
            ylim_rmse = [3,11]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=2)
        elif data_sets=="concrete":
            ylim_rmse = [6,10]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=1)
        elif data_sets=="energy":
            ylim_rmse = [1,10]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=3.0)
        elif data_sets=="kin8nm":
            ylim_rmse = [0.08,0.20]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=0.03)
        elif data_sets=="naval":
            ylim_rmse = [0,0.015]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=0.005)
        elif data_sets=="powerplant":
            ylim_rmse = [4.2,4.8]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=0.2)
        elif data_sets=="wine":
            ylim_rmse = [0.64,0.76]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=0.04)
        elif data_sets=="yacht":
            ylim_rmse = [1,11]
            yticks_rmse = np.arange(ylim_rmse[0],1e-5+ylim_rmse[1],step=2.0)
    
    
    fig = plt.figure(figsize=fig_size)
    plt.title(data_sets, fontsize = title_size)
    plt.xlabel('Epochs', fontdict = {'fontsize': label_size})
    plt.ylabel('Test RMSE', fontdict = {'fontsize': label_size})
    plt.plot(pepoch, prmsem, linewidth=lw, color = "green")
    plt.fill_between(pepoch, prmsem-prmses, prmsem+prmses, alpha = 0.3, color = "green")
    plt.plot(bepoch, brmsem, linewidth=lw, color = "black")
    plt.fill_between(bepoch, brmsem-brmses, brmsem+brmses, alpha = 0.3, color = "black")
    plt.plot(vepoch, vrmsem, linewidth=lw, color = "red")
    plt.fill_between(vepoch, vrmsem-vrmses, vrmsem+vrmses, alpha = 0.3, color = "red")
    plt.grid(True,which="both",color=grid_color, linewidth=grid_lw)
    plt.xticks(x_ticks,fontsize=tick_size)
    plt.xlim([0,40])
    if set_y_axis == True:
        plt.yticks(yticks_rmse,fontsize=tick_size)
        plt.ylim(ylim_rmse)
    else:
        plt.yticks(fontsize=tick_size)
    
    plt.ticklabel_format(axis='y', style='sci')
    ax = fig.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
    
    plt.tight_layout()
    plt.savefig(pdf_folder + "uci_rmse_" + data_sets + ".pdf", bbox_inches='tight')



###################
## Create legend ##
###################
    
fig = plt.figure(figsize=fig_size)
plt.plot(bepoch, brmsem, linewidth=lw, color = "black")
plt.plot(pepoch, prmsem, linewidth=lw, color = "green")
plt.plot(vepoch, vrmsem, linewidth=lw, color = "red")

legend = plt.legend(["BBVI", "Vprop", "Vadam"], loc = 'upper center', bbox_to_anchor=(2.3, 1.45), borderaxespad=0., ncol = 3, prop={'size': 14})

pad = 0
def export_legend(legend, filename="./plots/legend.pdf", expand=[-pad,-pad,pad,pad]):
   fig  = legend.figure
   fig.canvas.draw()
   bbox  = legend.get_window_extent()
   bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
   bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
   fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend, filename=pdf_folder + "uci_legend.pdf")



plt.ion()






































