% @Author: amishkin
% @Date:   2018-07-10T15:15:56-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T11:45:03-07:00

clear
methods = {'Vadam', 'VOGN'};
% specify the methods to run.
num_splits = 20;
dataset_ran = 'breast_cancer_scale';
M_settings = {[1, 8, 16, 32, 64], [1]};

all_Kls = [];

file_name = strcat('./breast_cancer_experiment_data/breast_cancer_scale_mf_exact_M_345_restart_1.mat');
exact = load(file_name);
vi_exact.sigma = exact.Sigma;
vi_exact.mu = exact.mu;

% Parse the experiment data from the logs.
for index = 1:length(methods)
    full_results = [];
    method_name = methods{index};
    M_list = M_settings{index};
    KLs = [];
    for m = 1:length(M_list)
        m_value = M_list(m);
        for split = 1:num_splits
            file_name = strcat('./breast_cancer_experiment_data/breast_cancer_scale_', method_name, '_M_', num2str(m_value) , '_restart_', num2str(split), '.mat');
            % Load the final variational distribution.
            method = load(file_name);
            method.sigma = method.Sigma;
            method.mu = method.mu;
            % compute metrics
            all_KLs(split,m,index) = (KL(vi_exact, method) + KL(method, vi_exact)) / 2; % symmetric KL to exact mean-field
        end
    end
end

% create plot 2c:
h3 = figure('Position', [50,150,800,800]);
s = tight_subplot(1,1, 0.02, [0.12, 0.05], [0.12, 0.02]); %create and get handle to the subplot axes
s1 = s(1);
ax1 = gca;

mean_line = line([0 10], [mean(all_KLs(:,1,2)), mean(all_KLs(:,1,2))]);
hold on
bx = boxplot(all_KLs(:,:,1), 'PlotStyle', 'Traditional', 'Notch','on', 'Labels', {'1','8', '16', '32', '64'});
set(bx, {'linew'}, {8})
grid on;

upper = refline(0, prctile(all_KLs(:,1,2), 75));
lower = refline(0, prctile(all_KLs(:,1,2), 25));

set(upper, 'color','blue', 'linestyle', '--', 'linewidth', 4);
set(lower, 'color','blue', 'linestyle', '--', 'linewidth', 4);
set(mean_line, 'color','black', 'linestyle', '-', 'linewidth', 4);

set(findall(gcf,'-property','FontSize'),'FontSize',38);
set(findobj(gcf,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
% tightfig();

% subp = get(h3,'children')

fontsize = 26;
set(ax1, 'fontsize', fontsize); %KL btm
ax1.XLabel.String = 'Minibatch size';
ax1.YLabel.String = 'Symmetric KL-divergence';
ax1.GridAlpha = 0.4;

ax1.XTick = [1 2 3 4 5];
ax1.XTickLabel = {'1', '8', '16', '32', '64'};
ax1.XLim = [0.5 5.5];

ax1.YLim = [0, 10.5];
ax1.YTick = [0, 2, 4, 6, 8, 10, 12];
ax1.YTickLabel = {'0', '2', '4', '6', '8', '10', '12'};

axesObjs = get(h3, 'Children');  %axes handles
dataObjs = get(axesObjs, 'Children'); %handles to low-level graphics objects in axes

vogn_line = dataObjs(4);
set(vogn_line,'color', 'b');
set(vogn_line,'linewidth', 6);
set(vogn_line,'linestyle', '-');

line_x = dataObjs(1);
set(line_x,'color', 'b');
set(line_x,'linewidth', 4);
set(line_x,'linestyle', '--');

line_xx = dataObjs(2);
set(line_xx,'color', 'b');
set(line_xx,'linewidth', 4);
set(line_xx,'linestyle', '--');

cell_obj = dataObjs(3);

boxes = get(cell_obj, 'Children'); %handles to low-level graphics objects in axes

boxes_outlier = boxes(1:5);
boxes_inner = boxes(6:10);
boxes_outer = boxes(11:15);
boxes_hor_low = boxes(16:20);
boxes_hor_up = boxes(21:25);
boxes_ver_low = boxes(26:30);
boxes_ver_up = boxes(31:35);

linewidth = 5;
alpha = 0.8;

for i = [1 : 5]
    set(boxes_outlier(i),'markeredgecolor', 'g');
    set(boxes_outlier(i),'marker', 'none');
    set(boxes_outlier(i),'linewidth', linewidth);

    set(boxes_inner(i),'color', 'r');
    set(boxes_inner(i),'linewidth', linewidth);

    set(boxes_outer(i),'color', [1,0,0, alpha]);
    set(boxes_outer(i),'linewidth', linewidth);

    set(boxes_hor_low(i),'color', 'k');
    set(boxes_hor_low(i),'linewidth', linewidth);

    set(boxes_hor_up(i),'color', 'k');
    set(boxes_hor_up(i),'linewidth', linewidth);

    set(boxes_ver_low(i),'color', 'k');
    set(boxes_ver_low(i),'linewidth', linewidth);

    set(boxes_ver_up(i),'color', 'k');
    set(boxes_ver_up(i),'linewidth', linewidth);
end

hold on;
dummy = plot(0, 0, '-', 'color', 'r', 'linewidth', 6);
leg = legend([dataObjs(4), dummy], {'VOGN-1', 'Vadam'}, 'Location', 'northwest');

leg.FontSize = 30;

f = h3;
set(f,'Units','Inches');
pos = get(f,'Position');
set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

savefig('figures/figure_two_c.fig');
saveas(f, 'figures/figure_two_c.pdf');

close all
