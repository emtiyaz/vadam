% @Author: amishkin
% @Date:   2018-06-07T18:38:11-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   amishkin
% @Last modified time: 2018-07-10T15:10:42-07:00



function [delta, algo_params] =  get_expt_params(dataset_name, method_name, trial_params);
    % this file returns expreiment parameters for each data, method pair

    % set experiment parameters
    mini_batch_size = trial_params(2);
    num_epochs = trial_params(3);
    switch dataset_name
    case {'murphy_synth'}
        delta = 100;
        data_set_size = 60;
    case {'breast_cancer_scale'}
        delta =  1;
        data_set_size = 341;
    case {'australian_scale'}
        data_set_size = 345;
        delta =  1e-5;
    case {'a1a'}
        data_set_size = 1605;
        delta =  2.8072;
    case {'usps_3vs5'}
        data_set_size = 770;
        delta =  25;
    case {'colon-cancer'}
        data_set_size = 31;
        delta =  596.3623;
    case {'covtype_binary_scale'}
        data_set_size = 290506;
        delta =  1e-5;
    otherwise
        error('no such dataset');
    end
    max_iter = floor((num_epochs * data_set_size) / mini_batch_size);

    switch method_name
    case {'mf_exact'}
        algo_params = struct(...
        'max_iters', num_epochs);
    otherwise
        algo_params = struct(...
        'max_iters', max_iter,...
        'num_epochs', num_epochs,...
        'beta', trial_params(4),...
        'alpha', trial_params(5), ...
        'decay_rate', trial_params(6), ...
        'num_samples', trial_params(1), ...
        'mini_batch_size', mini_batch_size);
    end
end
