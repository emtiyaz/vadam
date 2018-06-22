from bayesopt import run_bayesopt, run_final_vprop

####################
## Set parameters ##
####################

# Folder for storing results
results_folder = "./results/"

# Folder containing data
data_folder = "./../vadam/data"

# Data set
data_params = {'data_set': None,
               'n_splits': 5,
               'seed': 123}

# Model parameters
model_params = {'hidden_sizes': [50],
                'act_func': 'relu',
                'prior_prec': None,
                'noise_prec': None}

# Training parameters
train_params = {'num_epochs': 40,
                'batch_size': None,
                'train_mc_samples': None,
                'eval_mc_samples': 100,
                'seed': 123}

# Optimizer parameters
optim_params = {'learning_rate': 0.01,
                'betas': (0.9, 0.99),
                'prec_init': 10.0}

# Evaluations per epoch
evals_per_epoch = 1000

# BO parameters
bo_params = {'acq': 'ei',
             'init_points': 5, 
             'n_iter': 25}
length_scale = [1, 2]
nu = 2.5
alpha = 1e-2

#################
## Define grid ##
#################

all_data_sets = \
["yacht" + str(i) for i in range(20)] + \
["energy" + str(i) for i in range(20)] + \
["boston" + str(i) for i in range(20)] + \
["concrete" + str(i) for i in range(20)] + \
["wine" + str(i) for i in range(20)] + \
["kin8nm" + str(i) for i in range(20)] + \
["naval" + str(i) for i in range(20)] + \
["powerplant" + str(i) for i in range(20)]

large_data_sets = \
["wine" + str(i) for i in range(20)] + \
["kin8nm" + str(i) for i in range(20)] + \
["naval" + str(i) for i in range(20)] + \
["powerplant" + str(i) for i in range(20)]

grid = [(data_set) 
for data_set in all_data_sets]

#########################
## Run BO sequentially ##
#########################

for i, (data_set) in enumerate(grid):
    
    data_params['data_set'] = data_set
    
    if data_set in ["naval" + str(i) for i in range(20)] + ["powerplant" + str(i) for i in range(20)]:
        param_bounds = {'log_noise_prec': (1, 5), 
                        'log_prior_prec': (-3, 4)}
    else:
        param_bounds = {'log_noise_prec': (0, 5), 
                        'log_prior_prec': (-4, 4)}
        
    if data_set in large_data_sets:
        train_params['batch_size'] = 128
        train_params['train_mc_samples'] = 10
    else:
        train_params['batch_size'] = 32
        train_params['train_mc_samples'] = 20
    
    run_bayesopt(method = "bbb", 
                 data_folder = data_folder,
                 results_folder = results_folder, 
                 data_params = data_params, 
                 model_params = model_params, 
                 train_params = train_params, 
                 optim_params = optim_params, 
                 evals_per_epoch = evals_per_epoch, 
                 param_bounds = param_bounds, 
                 bo_params = bo_params, 
                 length_scale = length_scale, 
                 nu = nu, 
                 alpha = alpha)
    
    if data_set in large_data_sets:
        train_params['batch_size'] = 128
        train_params['train_mc_samples'] = 5
    else:
        train_params['batch_size'] = 32
        train_params['train_mc_samples'] = 10
        
    run_bayesopt(method = "vadam", 
                 data_folder = data_folder,
                 results_folder = results_folder, 
                 data_params = data_params, 
                 model_params = model_params, 
                 train_params = train_params, 
                 optim_params = optim_params, 
                 evals_per_epoch = evals_per_epoch, 
                 param_bounds = param_bounds, 
                 bo_params = bo_params, 
                 length_scale = length_scale, 
                 nu = nu, 
                 alpha = alpha)

    run_final_vprop(data_folder = data_folder,
                    results_folder = results_folder, 
                    data_params = data_params, 
                    model_params = model_params, 
                    train_params = train_params, 
                    optim_params = optim_params, 
                    evals_per_epoch = evals_per_epoch, 
                    param_bounds = param_bounds, 
                    bo_params = bo_params)


