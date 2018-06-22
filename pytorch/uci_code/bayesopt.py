import os
import math
import pickle
import copy
from sklearn.gaussian_process.kernels import Matern
from experiments_cv import CrossValExperimentVadamMLPReg, CrossValExperimentBBBMLPReg
from experiments import ExperimentVadamMLPReg, ExperimentBBBMLPReg, ExperimentVpropMLPReg
from bayes_opt import BayesianOptimization

#############################
## Define useful functions ##
#############################

def get_cv_average(list_of_dicts, key):
    value = 0
    for metdict in list_of_dicts:
        value += metdict[key][0]
    value = value/len(list_of_dicts)
    return value

def folder_name(experiment_name, param_bounds, bo_params, data_params, model_params, train_params, optim_params, results_folder="./results"):
    pp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(param_bounds.items()))[:-1]
    bp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(bo_params.items()))[:-1]
    dp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(data_params.items()))[:-1]
    mp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(model_params.items()))[:-1]
    tp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(train_params.items()))[:-1]
    op = ''.join('{}:{}|'.format(key, val) for key, val in sorted(optim_params.items()))[:-1]
    return os.path.join(results_folder, experiment_name, pp, bp, dp, mp, tp, op)



###############################################
## Define function for Bayesian optimization ##
###############################################

def run_bayesopt(method, data_folder, results_folder, data_params, model_params, train_params, optim_params, evals_per_epoch, param_bounds, bo_params, length_scale, nu, alpha):
    
    if method == "vadam":
        experiment_name = "bayesopt_vadam"
    elif method == "bbb":
        experiment_name = "bayesopt_bbb"
        
    gp_params = {'kernel': Matern(nu=nu, length_scale=length_scale),
                 'alpha': alpha}
        
    ###################################
    ## Define CV experiment function ##
    ###################################
    
    def cv_exp(log_noise_prec, log_prior_prec):
        
        model_params_cv = copy.deepcopy(model_params)
        model_params_cv['prior_prec'] = math.exp(log_prior_prec)
        model_params_cv['noise_prec'] = math.exp(log_noise_prec)
        
        #######################
        ## Define experiment ##
        #######################
        
        if method=="vadam":
            optim_params_cv = copy.deepcopy(optim_params)
            optim_params_cv['prec_init'] = model_params_cv['prior_prec']
            experiment = CrossValExperimentVadamMLPReg(data_folder = data_folder,
                                                       data_params = data_params, 
                                                       model_params = model_params_cv, 
                                                       train_params = train_params, 
                                                       optim_params = optim_params_cv,
                                                       normalize_x = True,
                                                       normalize_y = True)
        elif method=="bbb":
            experiment = CrossValExperimentBBBMLPReg(data_folder = data_folder,
                                                     data_params = data_params, 
                                                     model_params = model_params_cv, 
                                                     train_params = train_params, 
                                                     optim_params = optim_params,
                                                     normalize_x = True,
                                                     normalize_y = True)
        
        
        ####################
        ## Run experiment ##
        ####################
        
        experiment.run(log_metric_history = False)
        
        ##########################
        ## Rerturn Avg. Test LL ##
        ##########################
        
        logloss = get_cv_average(experiment.final_metric, key='test_pred_logloss')
        return -logloss

    ###############################
    ## Run Bayesian optimization ##
    ###############################
    
    bo = BayesianOptimization(cv_exp, param_bounds)
    
    bo.maximize(init_points=bo_params['init_points'], n_iter=bo_params['n_iter'], acq=bo_params['acq'], **gp_params)
    
    ######################
    ## Store BO results ##
    ######################
    
    folder = folder_name(results_folder = results_folder,
                         experiment_name = experiment_name, 
                         param_bounds = param_bounds, 
                         bo_params = bo_params, 
                         data_params = data_params, 
                         model_params = model_params, 
                         train_params = train_params, 
                         optim_params = optim_params)
    
    os.makedirs(folder, exist_ok=True)
    
    output = open(os.path.join(folder, 'res_all.pkl'), 'wb')
    pickle.dump(bo.res['all'], output)
    output.close()
    output = open(os.path.join(folder, 'res_max.pkl'), 'wb')
    pickle.dump(bo.res['max'], output)
    output.close()

    ############################################
    ## Run experiment with optimal parameters ##
    ############################################
        
    model_params_final = copy.deepcopy(model_params)
    model_params_final['prior_prec'] = math.exp(bo.res['max']['max_params']['log_prior_prec'])
    model_params_final['noise_prec'] = math.exp(bo.res['max']['max_params']['log_noise_prec'])
    
    #######################
    ## Define experiment ##
    #######################
    
    if method == "vadam":
        optim_params_final = copy.deepcopy(optim_params)
        optim_params_final['prec_init'] = model_params_final['prior_prec']
        experiment = ExperimentVadamMLPReg(results_folder = results_folder,
                                           data_folder = data_folder,
                                           data_set = data_params['data_set'], 
                                           model_params = model_params_final, 
                                           train_params = train_params, 
                                           optim_params = optim_params_final,
                                           evals_per_epoch = evals_per_epoch,
                                           normalize_x = True,
                                           normalize_y = True)
    elif method == "bbb":
        experiment = ExperimentBBBMLPReg(results_folder = results_folder,
                                         data_folder = data_folder,
                                         data_set = data_params['data_set'], 
                                         model_params = model_params_final, 
                                         train_params = train_params, 
                                         optim_params = optim_params,
                                         evals_per_epoch = evals_per_epoch,
                                         normalize_x = True,
                                         normalize_y = True)
    
    
    ####################
    ## Run experiment ##
    ####################
    
    experiment.run(log_metric_history = True)
    
    experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False,
                    folder_path = folder)



##############################################################################
## Define function for getting final Vprop run with settings found by Vadam ##
##############################################################################
    
def run_final_vprop(data_folder, results_folder, data_params, model_params, train_params, optim_params, evals_per_epoch, param_bounds, bo_params):
    
    ######################
    ## Store BO results ##
    ######################
    
    folder_bo = folder_name(results_folder = results_folder,
                         experiment_name = "bayesopt_vadam", 
                         param_bounds = param_bounds, 
                         bo_params = bo_params, 
                         data_params = data_params, 
                         model_params = model_params, 
                         train_params = train_params, 
                         optim_params = optim_params)
    
    
    optim_params_vprop = copy.deepcopy(optim_params)
    beta = optim_params_vprop['betas'][1]
    optim_params_vprop.pop('betas', None)
    optim_params_vprop['beta'] = beta
    
    folder = folder_name(results_folder = results_folder,
                         experiment_name = "final_vprop", 
                         param_bounds = param_bounds, 
                         bo_params = bo_params, 
                         data_params = data_params, 
                         model_params = model_params, 
                         train_params = train_params, 
                         optim_params = optim_params_vprop)
    
    output = open(os.path.join(folder_bo, 'res_max.pkl'), 'rb')
    res_max = pickle.load(output)
    output.close()
    
    ############################################
    ## Run experiment with optimal parameters ##
    ############################################
    
    model_params_final = copy.deepcopy(model_params)
    model_params_final['prior_prec'] = math.exp(res_max['max_params']['log_prior_prec'])
    model_params_final['noise_prec'] = math.exp(res_max['max_params']['log_noise_prec'])
    optim_params_vprop['prec_init'] = model_params_final['prior_prec']
    
    #######################
    ## Define experiment ##
    #######################
    
    experiment = ExperimentVpropMLPReg(results_folder = results_folder,
                                       data_folder = data_folder,
                                       data_set = data_params['data_set'], 
                                       model_params = model_params_final, 
                                       train_params = train_params, 
                                       optim_params = optim_params_vprop,
                                       evals_per_epoch = evals_per_epoch,
                                       normalize_x = True,
                                       normalize_y = True)
    
    
    ####################
    ## Run experiment ##
    ####################
    
    experiment.run(log_metric_history = True)
    
    experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False,
                    folder_path = folder)