import os
import pickle

import numpy as np
from scipy import stats
    
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

for data_sets in ds:
    
    # Folder containing results
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
    bll = np.zeros([len(grid_marginalize)])
    brmse = np.zeros([len(grid_marginalize)])
    
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
        
        pkl_file = open(os.path.join(folder, 'final_metric.pkl'), 'rb')
        final_metric = pickle.load(pkl_file)
        pkl_file.close()
        
        bll[i] = final_metric['test_pred_logloss'][-1]
        brmse[i] = final_metric['test_pred_rmse'][-1]
    
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
    vll = np.zeros([len(grid_marginalize)])
    vrmse = np.zeros([len(grid_marginalize)])
    
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
        
        pkl_file = open(os.path.join(folder, 'final_metric.pkl'), 'rb')
        final_metric = pickle.load(pkl_file)
        pkl_file.close()
        
        vll[i] = final_metric['test_pred_logloss'][-1]
        vrmse[i] = final_metric['test_pred_rmse'][-1]
    
    
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
    
    pval_test = 0.01
    n_decimals = 2
    fmt_str = '{:0.' + str(n_decimals) + 'f}'
    
    ##############################
    ## t-test for log-liklihood ##
    ##############################
    
    if vllm > bllm:
        # compare to vadam
        _, p_value_b = stats.ttest_rel(vll, bll)
        
        sig_b = p_value_b <= pval_test
        
        print("\n\n\n", data_sets, ", Vadam is best with ll = ", fmt_str.format(vllm), "$\pm$", fmt_str.format(vlls))
        if sig_b:
            print("BBVI is significantly worse with ll = ", fmt_str.format(bllm), "$\pm$", fmt_str.format(blls), ", pval = ", p_value_b)
        else:
            print("BBVI is comparable with ll = ", fmt_str.format(bllm), "$\pm$", fmt_str.format(blls), ", pval = ", p_value_b)
        
    elif bllm > vllm:
        # compare to bbvi
        _, p_value_v = stats.ttest_rel(bll, vll)
        
        sig_v = p_value_v <= pval_test
        
        print("\n\n\n", data_sets, ", BBVI is best with ll = ", fmt_str.format(bllm), "$\pm$", fmt_str.format(blls))
        if sig_v:
            print("Vadam is significantly worse with ll = ", fmt_str.format(vllm), "$\pm$", fmt_str.format(vlls), ", pval = ", p_value_v)
        else:
            print("Vadam is comparable with ll = ", fmt_str.format(vllm), "$\pm$", fmt_str.format(vlls), ", pval = ", p_value_v)
        
    else:
        print("error!")
    
    #####################
    ## t-test for rmse ##
    #####################
    
    if vrmsem < brmsem :
        # compare to vadam
        _, p_value_b = stats.ttest_rel(vrmse, brmse)
        
        sig_b = p_value_b <= pval_test
        
        print("\n", data_sets, ", Vadam is best with rmse = ", fmt_str.format(vrmsem), "$\pm$", fmt_str.format(vrmses))
        if sig_b:
            print("BBVI is significantly worse with rmse = ", fmt_str.format(brmsem), "$\pm$", fmt_str.format(brmses), ", pval = ", p_value_b)
        else:
            print("BBVI is comparable with rmse = ", fmt_str.format(brmsem), "$\pm$", fmt_str.format(brmses), ", pval = ", p_value_b)
        
    elif brmsem < vrmsem:
        # compare to bbvi
        _, p_value_v = stats.ttest_rel(brmse, vrmse)
        
        sig_v = p_value_v <= pval_test
        
        print("\n", data_sets, ", BBVI is best with rmse = ", fmt_str.format(brmsem), "$\pm$", fmt_str.format(brmses))
        if sig_v:
            print("Vadam is significantly worse with rmse = ", fmt_str.format(vrmsem), "$\pm$", fmt_str.format(vrmses), ", pval = ", p_value_v)
        else:
            print("Vadam is comparable with rmse = ", fmt_str.format(vrmsem), "$\pm$", fmt_str.format(vrmses), ", pval = ", p_value_v)
        
    else:
        print("error!")
    