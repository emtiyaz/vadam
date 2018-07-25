% @Author: amishkin
% @Date:   2018-07-10T14:52:36-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T11:47:43-07:00

% Load the dataset with a fixed seed.
addpath(genpath('.'))

output_dir = './usps_experiment_data'
mkdir(output_dir)

dataset_name = 'usps_3vs5';
[y, X, y_te, X_te] = get_data_log_reg(dataset_name, 1);
[N,D] = size(X);

% Set initial conditions.
mu_start = zeros(D,1);
s = 1.*ones(D,1);
sigma_start = diag(1./s);

% Run Vadam with different minibatch sizes:
method = 'Vadam'

num_samples = 1
beta = 0.01;
alpha = 0.01;
decay_rate = 0.55

num_restarts = 20;

%  List batch sizes and epochs for each batch size.
M_list = [64];
epoch_list = [10000];

run_experiment(dataset_name, method, M_list, epoch_list, alpha, beta, decay_rate, num_samples, num_restarts, mu_start, sigma_start, 1, output_dir)

% Run VOGN with a minibatch size of one:
method = 'VOGN'
num_samples = 1
beta = 0.0005;
alpha = 0.0005;
decay_rate = 0;

num_restarts = 20;

%  List batch sizes and epochs for each batch size.
M_list = [1];
epoch_list = [200];

run_experiment(dataset_name, method, M_list, epoch_list, alpha, beta, decay_rate, num_samples, num_restarts, mu_start, sigma_start, 1, output_dir)

% Run MF_Exact
method = 'mf_exact'
num_restarts = 20;

%  List batch sizes and epochs for each batch size.
epoch_list = [500];

run_experiment(dataset_name, method, M_list, epoch_list, 0, 0, 0, 0, num_restarts, mu_start, sigma_start, 1, output_dir)


% generate figure 2(b) using the data from this experiment:
make_fig_two_b
