% @Author: amishkin
% @Date:   2018-07-10T15:23:29-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T11:47:45-07:00

methods_ran = {'mf_exact', 'VOGN', 'Vadam'}
% specify the methods to run.
num_splits = 20;
dataset_ran = 'usps_3vs5';
M_settings = [1, 1, 64];

all_kls = [];
all_nlZs = [];
all_lls = [];

for index = 1:length(methods_ran)
    index
    full_results = [];
    method_name = methods_ran{index}
    M = M_settings(index)

    KLs = [];
    nlZs = [];
    lls = [];

    for split = 1:num_splits
        file_name = strcat('./usps_experiment_data/usps_3vs5_mf_exact_M_1_restart_', num2str(split), '.mat');
        exact = load(file_name);
        vi_exact.sigma = exact.Sigma;
        vi_exact.mu = exact.mu;

        file_name = strcat('./usps_experiment_data/usps_3vs5_', method_name, '_M_', num2str(M),'_restart_', num2str(split), '.mat');

        method = load(file_name);
        method.sigma = method.Sigma;
        method.mu = method.mu;

        KLs(split) = (KL(vi_exact, method) + KL(method, vi_exact)) / 2;
        nlZs(split) = method.nlZ(end);
        lls(split) = method.log_loss(end);
    end

    all_kls = [all_kls; KLs];
    all_nlZs = [all_nlZs; nlZs];
    all_lls = [all_lls; lls];
end

plot_name = strcat('./logistic_regression_baselines/boxplots', dataset_ran, '_symmetric')
[y, X, y_te, X_te] = get_data_log_reg(dataset_ran, 1);
[N, D] = size(X);

all_nlZs = all_nlZs ./ N;
all_lls = all_lls ./ log2(exp(1));

f1 = gen_box_plot(all_nlZs', strcat(plot_name, 'nlZ_boxplot'), 'ELBO', strrep(dataset_ran, '_', ' '), j == 1, 0, 0);
f2 = gen_box_plot(all_lls', strcat(plot_name, 'll_boxplot'), 'LogLoss', strrep(dataset_ran, '_', ' '), j == 1, 0, 0);
f3 = gen_box_plot(all_kls', strcat(plot_name, 'KL_boxplot'), 'KL', strrep(dataset_ran, '_', ' '), j == 1, 1, 0);


f = combinator(f1, f2, f3);

set(f,'Units','Inches');
pos = get(f,'Position');
set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

savefig('figures/figure_two_b.fig')
saveas(f, 'figures/figure_two_b.pdf')
close all
