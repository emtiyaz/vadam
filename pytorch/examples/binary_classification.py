####################
## Set parameters ##
####################

# Data set
data_set = "australian_presplit"

# Model parameters
act_func = "relu"
hidden_sizes = [64]
prior_prec = 1.0

# Training parameters
num_epochs = 200
batch_size = 128
seed = 123

# Optimizer parameters
optimizer = "vadam" # choose between "adam" and"vadam"
learning_rate = 0.005
betas = (0.9,0.995)

# VI parameters
train_mc_samples = 20
eval_mc_samples = 20
prec_init = 1.0

################
## Check CUDA ##
################

import torch

use_cuda = torch.cuda.is_available()

#####################
## Set random seed ##
#####################

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


###############
## Load data ##
###############

from vadam.datasets import Dataset

data = Dataset(data_set = data_set,
               data_folder = "./../vadam/data")
train_loader = data.get_train_loader(batch_size)

x_train, y_train = data.load_full_train_set(use_cuda = use_cuda)
x_test, y_test = data.load_full_test_set(use_cuda = use_cuda)

##################
## Define model ##
##################

from vadam.models import MLP

model = MLP(input_size = data.num_features,
            hidden_sizes = hidden_sizes,
            output_size = None,
            act_func = act_func)
if use_cuda:
    model = model.cuda()

#######################
## Define prediction ##
#######################

def prediction(x):
    logits = model(x)
    return logits

######################
## Define objective ##
######################

from vadam.metrics import avneg_loglik_bernoulli

objective = avneg_loglik_bernoulli


##############
## Use Adam ##
##############

if optimizer=="adam":
    
    # Use the Adam optimizer for MLE

    from torch.optim import Adam
    from vadam.metrics import avneg_loglik_bernoulli, sigmoid_accuracy
    
    optimizer = Adam(model.parameters(),
                     lr = learning_rate,
                     betas = betas,
                     weight_decay = prior_prec / data.get_train_size())
    
    # Evaluate using the point estimate
    
    metric_history = dict(train_logloss=[], train_accuracy=[],
                          test_logloss=[], test_accuracy=[])
    
    def evaluate_model(x_train, y_train, x_test, y_test):
    
        # Store train metrics
        logits = model(x_train)
        metric_history['train_logloss'].append(avneg_loglik_bernoulli(logits, y_train).detach().cpu().item())
        metric_history['train_accuracy'].append(sigmoid_accuracy(logits, y_train).detach().cpu().item())
        
        # Store test metrics
        logits = model(x_test)
        metric_history['test_logloss'].append(avneg_loglik_bernoulli(logits, y_test).detach().cpu().item())
        metric_history['test_accuracy'].append(sigmoid_accuracy(logits, y_test).detach().cpu().item())

elif optimizer=="vadam":
    
    # Use the Vadam optimizer for VI

    from vadam.optimizers import Vadam
    from vadam.metrics import predictive_avneg_loglik_bernoulli, sigmoid_predictive_accuracy
    
    optimizer = Vadam(model.parameters(),
                      lr = learning_rate,
                      betas = betas,
                      prior_prec = prior_prec,
                      prec_init = prec_init,
                      train_set_size = data.get_train_size())
    
    # Evaluate using the predictive distribution
    
    metric_history = dict(train_logloss=[], train_accuracy=[],
                          test_logloss=[], test_accuracy=[])
        
    def evaluate_model(x_train, y_train, x_test, y_test):
    
        # Store train metrics
        logits_list = optimizer.get_mc_predictions(model.forward, inputs = x_train, mc_samples = eval_mc_samples, ret_numpy=False)
        metric_history['train_logloss'].append(predictive_avneg_loglik_bernoulli(logits_list, y_train).detach().cpu().item())
        metric_history['train_accuracy'].append(sigmoid_predictive_accuracy(logits_list, y_train).detach().cpu().item())
        
        # Store test metrics
        logits_list = optimizer.get_mc_predictions(model.forward, inputs = x_test, mc_samples = eval_mc_samples, ret_numpy=False)
        metric_history['test_logloss'].append(predictive_avneg_loglik_bernoulli(logits_list, y_test).detach().cpu().item())
        metric_history['test_accuracy'].append(sigmoid_predictive_accuracy(logits_list, y_test).detach().cpu().item())


#####################
## Define printing ##
#####################

def print_progress(epoch):

    # Print progress
    print('Epoch [{}/{}], Logloss: {:.4f}, Test Logloss: {:.4f}'.format(
            epoch+1,
            num_epochs,
            metric_history['train_logloss'][-1],
            metric_history['test_logloss'][-1]))

#################
## Train model ##
#################

for epoch in range(num_epochs):

    # Set model in training mode
    model.train(True)

    for i, (x, y) in enumerate(train_loader):

        # Prepare minibatch
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        # Update parameters
        def closure():
            optimizer.zero_grad()
            logits = prediction(x)
            loss = objective(logits, y)
            loss.backward()
            return loss
        loss = optimizer.step(closure)

    # Set model in test mode
    model.train(False)

    # Evaluate model
    evaluate_model(x_train, y_train, x_test, y_test)

    # Print progress
    print_progress(epoch)


#######################
## Visualize results ##
#######################

import matplotlib.pyplot as plt

# Plot logloss
plt.figure()
plt.plot(metric_history['train_logloss'], 'b-')
plt.plot(metric_history['test_logloss'], 'r-')
plt.legend(["Train", "Test"])
plt.grid(True,which="both",color='0.75')
plt.xlabel('Epoch')
plt.ylabel('Logloss')

# Plot accuracy
plt.figure()
plt.plot(metric_history['train_accuracy'], 'b-')
plt.plot(metric_history['test_accuracy'], 'r-')
plt.legend(["Train", "Test"])
plt.grid(True,which="both",color='0.75')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Show plot
plt.show()