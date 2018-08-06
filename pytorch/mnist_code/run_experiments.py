from experiments import ExperimentVadamMLPClass, ExperimentBBBMLPClass

####################
## Set parameters ##
####################

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
                'train_mc_samples': None,
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

#################
## Define grid ##
#################

grid = [(hidden_sizes, mc, bs, prec) 
for hidden_sizes in ([400], [400,400])
for mc in (1, 10)
for bs in (1, 10, 100)
for prec in (1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2)]

####################################
## Run experiemtents sequentially ##
####################################

for i, (hidden_sizes, mc, bs, prec) in enumerate(grid):

    model_params['hidden_sizes'] = hidden_sizes
    model_params['prior_prec'] = prec
    optim_params['prec_init'] = prec
    optim_params_vogn['prec_init'] = prec
    train_params['train_mc_samples'] = mc
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
    
    # Run Vadam
    experiment = ExperimentVadamMLPClass(results_folder = results_folder, 
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
    experiment = ExperimentBBBMLPClass(results_folder = results_folder, 
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
