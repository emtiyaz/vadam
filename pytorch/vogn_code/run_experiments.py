from experiments import ExperimentVadamMLPBinClass, ExperimentBBBMLPBinClass, ExperimentVOGNMLPBinClass

####################
## Set parameters ##
####################

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
                'train_mc_samples': None,
                'eval_mc_samples': 10,
                'seed': 123}

# Optimizer parameters
optim_params = {'learning_rate': 0.001,
                'betas': (0.9, 0.999),
                'prec_init': 1.0}
optim_params_vogn = {'learning_rate': 0.001,
                     'beta': 0.999,
                     'prec_init': 1.0}

# Evaluations per epoch
evals_per_epoch = 1000

#################
## Define grid ##
#################

grid = [(bs, mc) 
for bs in (1, 4, 16, 128)
for mc in (1, 16)]

####################################
## Run experiemtents sequentially ##
####################################

for i, (bs, mc) in enumerate(grid):

    train_params['train_mc_samples'] = mc
    train_params['batch_size'] = bs
    train_params['num_epochs'] = 16 * bs
    
    # Run Vadam
    experiment = ExperimentVadamMLPBinClass(results_folder = results_folder, 
                                            data_folder = data_folder,
                                            data_set = data_set, 
                                            model_params = model_params, 
                                            train_params = train_params, 
                                            optim_params = optim_params,
                                            evals_per_epoch = evals_per_epoch,
                                            normalize_x = False)
    
    experiment.run(log_metric_history = True)
    
    experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False)
    
    # Run BBVI
    experiment = ExperimentBBBMLPBinClass(results_folder = results_folder, 
                                          data_folder = data_folder,
                                          data_set = data_set, 
                                          model_params = model_params, 
                                          train_params = train_params, 
                                          optim_params = optim_params,
                                          evals_per_epoch = evals_per_epoch,
                                          normalize_x = False)
    
    experiment.run(log_metric_history = True)
    
    experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False)
        
    # Run VOGN
    experiment = ExperimentVOGNMLPBinClass(results_folder = results_folder, 
                                           data_folder = data_folder,
                                           data_set = data_set, 
                                           model_params = model_params, 
                                           train_params = train_params, 
                                           optim_params = optim_params_vogn,
                                           evals_per_epoch = evals_per_epoch,
                                           normalize_x = False)
    
    experiment.run(log_metric_history = True)
    
    experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False)
