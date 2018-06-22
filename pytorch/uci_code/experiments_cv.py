import os
import pickle

import torch
import numpy as np

from vadam.datasets import DatasetCV, DEFAULT_DATA_FOLDER
from vadam.models import MLP, BNN
import vadam.metrics as metrics

from torch.optim import Adam
from vadam.optimizers import Vadam

###############################################################################
## Define function that specify folder naming convention for storing results ##
###############################################################################

def folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder="./results"):
    dp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(data_params.items()))[:-1]
    mp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(model_params.items()))[:-1]
    tp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(train_params.items()))[:-1]
    op = ''.join('{}:{}|'.format(key, val) for key, val in sorted(optim_params.items()))[:-1]
    return os.path.join(results_folder, experiment_name, dp, mp, tp, op)



###############################################################
## Define function for loading only history from experiments ##
###############################################################

def load_final_metric(experiment_name, data_params, model_params, train_params, optim_params, results_folder="./results", silent_fail=False):
    
    # Folder to load history from
    folder = folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder)
    file = os.path.join(folder, 'final_metric.pkl')
    
    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, 'rb')
    final_metric = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return final_metric

def load_metric_history(experiment_name, data_params, model_params, train_params, optim_params, results_folder="./results", silent_fail=False):

    # Folder to load history from
    folder = folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder)
    file = os.path.join(folder, 'metric_history.pkl')
    
    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, 'rb')
    metric_history = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return metric_history

def load_objective_history(experiment_name, data_params, model_params, train_params, optim_params, results_folder="./results", silent_fail=False):

    # Folder to load history from
    folder = folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder)
    file = os.path.join(folder, 'objective_history.pkl')
    
    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, 'rb')
    objective_history = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return objective_history



######################################
## Define abstract experiment class ##
######################################

class CrossValExperiment():

    def __init__(self, data_params, model_params, train_params, optim_params, normalize_x=False, normalize_y=False, results_folder="./results", data_folder=DEFAULT_DATA_FOLDER, use_cuda=torch.cuda.is_available()):

        # Store parameters
        self.data_params = data_params
        self.model_params = model_params
        self.train_params = train_params
        self.optim_params = optim_params
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.use_cuda = use_cuda

        # Set random seed
        seed = train_params['seed']
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

        # Initialize metric history
        self.objective_history = [[] for _ in range(data_params['n_splits'])]

        # Initialize data
        self.data = DatasetCV(data_set = data_params['data_set'],
                              n_splits = data_params['n_splits'],
                              seed = data_params['seed'],
                              data_folder = data_folder)

        ## All subclasses should override:

        # Define folder name for results
        self.folder_name = None

        # Define prediction function
        self.prediction = None

        # Define objective
        self.objective = None

        # Initialize model
        self.model = None

        # Initialize optimizer
        self.optimizer = None

        # Initialize metric history
        self.metric_history = None

        # Initialize final metric
        self.final_metric = None

    def run(self, log_metric_history=True):

        # Prepare
        n_splits = self.data_params['n_splits']
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        seed = self.train_params['seed']

        # Set random seed
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)
            
        for split in range(n_splits):
            
            # Set current split
            self.data.set_current_split(split)

            # Prepare data loader for current split for training
            train_loader = self.data.get_current_train_loader(batch_size = batch_size)
    
            # Load full data set for evaluation
            x_train, y_train = self.data.load_current_train_set(use_cuda = self.use_cuda)
            x_val, y_val = self.data.load_current_val_set(use_cuda = self.use_cuda)
        
            # Compute normalization of x
            if self.normalize_x:
                self.x_means = torch.mean(x_train, dim=0)
                self.x_stds = torch.std(x_train, dim=0)
                self.x_stds[self.x_stds == 0] = 1
            
            # Compute normalization of y
            if self.normalize_y:
                self.y_mean = torch.mean(y_train)
                self.y_std = torch.std(y_train)
                if self.y_std==0:
                    self.y_std = 1
            
            # Initialize model
            self._init_model()
            
            # Initialize optimizer
            self._init_optimizer()
    
            # Train model
            for epoch in range(num_epochs):
    
                # Set model in training mode
                self.model.train(True)
            
                # Initialize batch objective accumulator
                batch_objective = []
    
                for i, (x, y) in enumerate(train_loader):
    
                    # Prepare minibatch
                    if self.use_cuda:
                        x, y = x.cuda(), y.cuda()
                
                    # Normalize x and y
                    if self.normalize_x:
                        x = (x-self.x_means)/self.x_stds
                    if self.normalize_y:
                        y = (y-self.y_mean)/self.y_std
    
                    # Update parameters
                    def closure():
                        self.optimizer.zero_grad()
                        logits = self.prediction(x)
                        loss = self.objective(logits, y)
                        loss.backward()
                        return loss
                    loss = self.optimizer.step(closure)

                    # Store batch objective
                    batch_objective.append(loss.detach().cpu().item())
                    
                # Compute and store average objective from last epoch
                self.objective_history[split].append(np.mean(batch_objective))
                    
                if log_metric_history:
                    
                    # Set model in test mode
                    self.model.train(False)
        
                    # Evaluate model
                    with torch.no_grad():
                        self._evaluate_model(self.metric_history[split], x_train, y_train, x_val, y_val)
        
                    # Print progress
                    self._print_progress(split, epoch)
                
                else:
                    
                    # Print average objective from last epoch
                    self._print_objective(split, epoch)
            
            # Set model in test mode
            self.model.train(False)
    
            # Evaluate model
            with torch.no_grad():
                self._evaluate_model(self.final_metric[split], x_train, y_train, x_val, y_val)

    def _init_model(self):

        ## All subclasses should override:
        raise NotImplementedError

    def _init_optimizer(self):

        ## All subclasses should override:
        raise NotImplementedError

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        ## All subclasses should override:
        raise NotImplementedError

    def _print_progress(self, split, epoch):

        ## All subclasses should override:
        raise NotImplementedError

    def _print_objective(self, split, epoch):

        # Print average objective from last epoch
        print('Split [{}/{}], Epoch [{}/{}], Objective: {:.4f}'.format(
                split+1,
                self.data_params['n_splits'],
                epoch+1,
                self.train_params['num_epochs'],
                self.objective_history[split][-1]))

    def save(self, save_final_metric=True, save_metric_history=True, save_objective_history=True, create_folder=True, folder_path=None):

        # Define folder path
        if not folder_path:
            folder_path = self.folder_name
            
        # Create folder
        if create_folder:
            os.makedirs(folder_path, exist_ok=True)

        # Store history
        if save_final_metric:
            output = open(os.path.join(folder_path, 'final_metric.pkl'), 'wb')
            pickle.dump(self.final_metric, output)
            output.close()
        if save_metric_history:
            output = open(os.path.join(folder_path, 'metric_history.pkl'), 'wb')
            pickle.dump(self.metric_history, output)
            output.close()
        if save_objective_history:
            output = open(os.path.join(folder_path, 'objective_history.pkl'), 'wb')
            pickle.dump(self.objective_history, output)
            output.close()
            
    def load(self, load_final_metric=True, load_metric_history=True, load_objective_history=True, folder_path=None):

        # Define folder path
        if not folder_path:
            folder_path = self.folder_name
            
        # Load history
        if load_final_metric:
            pkl_file = open(os.path.join(folder_path, 'final_metric.pkl'), 'rb')
            self.final_metric = pickle.load(pkl_file)
            pkl_file.close()
        if load_metric_history:
            pkl_file = open(os.path.join(folder_path, 'metric_history.pkl'), 'rb')
            self.metric_history = pickle.load(pkl_file)
            pkl_file.close()
        if load_objective_history:
            pkl_file = open(os.path.join(folder_path, 'objective_history.pkl'), 'rb')
            self.objective_history = pickle.load(pkl_file)
            pkl_file.close()
            
            
            
#######################################
## Define experiment class for Vadam ##
#######################################

class CrossValExperimentVadamMLPReg(CrossValExperiment):

    def __init__(self, data_params, model_params, train_params, optim_params, normalize_x=False, normalize_y=False, results_folder="./results", data_folder=DEFAULT_DATA_FOLDER, use_cuda=torch.cuda.is_available()):
        super(type(self), self).__init__(data_params, model_params, train_params, optim_params, normalize_x, normalize_y, results_folder, data_folder, use_cuda)

        # Define name for experiment class
        experiment_name = "vadam_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder)

        # Define prediction function
        def prediction(x):
            logits = self.model(x)
            return logits
        self.prediction = prediction

        # Define objective
        def objective(mu, y):
            return metrics.avneg_loglik_gaussian(mu, y, tau = self.model_params['noise_prec'])
        self.objective = objective

        # Initialize metric history
        self.metric_history = [dict(elbo_neg_ave = [],
                                    train_pred_logloss=[], train_pred_rmse=[],
                                    test_pred_logloss=[], test_pred_rmse=[]) for _ in range(data_params['n_splits'])]
        
        # Initialize final metric
        self.final_metric = [dict(elbo_neg_ave = [],
                                  train_pred_logloss=[], train_pred_rmse=[],
                                  test_pred_logloss=[], test_pred_rmse=[]) for _ in range(data_params['n_splits'])]
    
    def _init_model(self):
        self.model = MLP(input_size = self.data.num_features,
                         hidden_sizes = self.model_params['hidden_sizes'],
                         output_size = self.data.num_classes,
                         act_func = self.model_params['act_func'])
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def _init_optimizer(self):
        self.optimizer = Vadam(self.model.parameters(),
                               lr = self.optim_params['learning_rate'],
                               betas = self.optim_params['betas'],
                               prior_prec = self.model_params['prior_prec'],
                               prec_init = self.optim_params['prec_init'],
                               num_samples = self.train_params['train_mc_samples'],
                               train_set_size = self.data.get_current_train_size())


    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):
        
        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params['noise_prec'] / (self.y_std**2)
        else:
            tau = self.model_params['noise_prec']
        
        # Normalize train x
        if self.normalize_x:
            x_train = (x_train-self.x_means)/self.x_stds
        
        # Get train predictions
        mu_list = self.optimizer.get_mc_predictions(self.model.forward, inputs = x_train, mc_samples = self.train_params['eval_mc_samples'], ret_numpy=False)
        
        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict['train_pred_logloss'].append(metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau = tau).detach().cpu().item())
        metric_dict['train_pred_rmse'].append(metrics.predictive_rmse(mu_list, y_train).detach().cpu().item())
        metric_dict['elbo_neg_ave'].append(metrics.avneg_elbo_gaussian(mu_list, y_train, tau = tau, train_set_size = self.data.get_current_train_size(), kl = self.optimizer.kl_divergence()).detach().cpu().item())

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test-self.x_means)/self.x_stds
        
        # Get test predictions
        mu_list = self.optimizer.get_mc_predictions(self.model.forward, inputs = x_test, mc_samples = self.train_params['eval_mc_samples'], ret_numpy=False)
        
        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]
        
        # Store test metrics
        metric_dict['test_pred_logloss'].append(metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau = tau).detach().cpu().item())
        metric_dict['test_pred_rmse'].append(metrics.predictive_rmse(mu_list, y_test).detach().cpu().item())

    def _print_progress(self, split, epoch):

        # Print progress
        print('Split [{}/{}], Epoch [{}/{}], Neg. Ave. ELBO: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f}'.format(
                split+1,
                self.data_params['n_splits'],
                epoch+1,
                self.train_params['num_epochs'],
                self.metric_history[split]['elbo_neg_ave'][-1],
                self.metric_history[split]['train_pred_logloss'][-1],
                self.metric_history[split]['test_pred_logloss'][-1]))
        
        
        
######################################
## Define experiment class for BBVI ##
######################################

class CrossValExperimentBBBMLPReg(CrossValExperiment):

    def __init__(self, data_params, model_params, train_params, optim_params, normalize_x=False, normalize_y=False, results_folder="./results", data_folder=DEFAULT_DATA_FOLDER, use_cuda=torch.cuda.is_available()):
        super(type(self), self).__init__(data_params, model_params, train_params, optim_params, normalize_x, normalize_y, results_folder, data_folder, use_cuda)

        # Define name for experiment class
        experiment_name = "bbb_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(experiment_name, data_params, model_params, train_params, optim_params, results_folder)

        # Define prediction function
        def prediction(x):
            mu_list = [self.model(x) for _ in range(self.train_params['train_mc_samples'])]
            return mu_list
        self.prediction = prediction

        # Define objective
        def objective(mu_list, y):
            return metrics.avneg_elbo_gaussian(mu_list, y, tau = self.model_params['noise_prec'], train_set_size = self.data.get_current_train_size(), kl = self.model.kl_divergence())
        self.objective = objective

        # Initialize metric history
        self.metric_history = [dict(elbo_neg_ave = [],
                                    train_pred_logloss=[], train_pred_rmse=[],
                                    test_pred_logloss=[], test_pred_rmse=[]) for _ in range(data_params['n_splits'])]
        
        # Initialize final metric
        self.final_metric = [dict(elbo_neg_ave = [],
                                  train_pred_logloss=[], train_pred_rmse=[],
                                  test_pred_logloss=[], test_pred_rmse=[]) for _ in range(data_params['n_splits'])]
    
    def _init_model(self):
        self.model = BNN(input_size = self.data.num_features,
                         hidden_sizes = self.model_params['hidden_sizes'],
                         output_size = self.data.num_classes,
                         act_func = self.model_params['act_func'],
                         prior_prec = self.model_params['prior_prec'],
                         prec_init = self.optim_params['prec_init'])
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def _init_optimizer(self):
        self.optimizer = Adam(self.model.parameters(),
                              lr = self.optim_params['learning_rate'],
                              betas = self.optim_params['betas'],
                              eps = 1e-8)


    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):
        
        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params['noise_prec'] / (self.y_std**2)
        else:
            tau = self.model_params['noise_prec']
        
        # Normalize train x
        if self.normalize_x:
            x_train = (x_train-self.x_means)/self.x_stds
        
        # Get train predictions
        mu_list = [self.model(x_train) for _ in range(self.train_params['eval_mc_samples'])]
        
        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict['train_pred_logloss'].append(metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau = tau).detach().cpu().item())
        metric_dict['train_pred_rmse'].append(metrics.predictive_rmse(mu_list, y_train).detach().cpu().item())
        metric_dict['elbo_neg_ave'].append(metrics.avneg_elbo_gaussian(mu_list, y_train, tau = tau, train_set_size = self.data.get_current_train_size(), kl = self.model.kl_divergence()).detach().cpu().item())

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test-self.x_means)/self.x_stds
        
        # Get test predictions
        mu_list = [self.model(x_test) for _ in range(self.train_params['eval_mc_samples'])]
        
        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]
        
        # Store test metrics
        metric_dict['test_pred_logloss'].append(metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau = tau).detach().cpu().item())
        metric_dict['test_pred_rmse'].append(metrics.predictive_rmse(mu_list, y_test).detach().cpu().item())

    def _print_progress(self, split, epoch):

        # Print progress
        print('Split [{}/{}], Epoch [{}/{}], Neg. Ave. ELBO: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f}'.format(
                split+1,
                self.data_params['n_splits'],
                epoch+1,
                self.train_params['num_epochs'],
                self.metric_history[split]['elbo_neg_ave'][-1],
                self.metric_history[split]['train_pred_logloss'][-1],
                self.metric_history[split]['test_pred_logloss'][-1]))