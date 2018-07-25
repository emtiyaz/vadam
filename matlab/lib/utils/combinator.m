% @Author: aaronmishkin
% @Date:   2018-06-12T18:19:10-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T11:51:06-07:00

function [f] = combinator(f1, f2, f3)

    ax1 = f1.CurrentAxes;
    ax2 = f2.CurrentAxes;
    ax3 = f3.CurrentAxes;

    h3 = figure(4);

    s = tight_subplot(3,1, 0.02, [0.065, 0.02], [0.15, 0.02]); %create and get handle to the subplot axes

    s1 = s(1);
    s2 = s(2);
    s3 = s(3);

    fig1 = get(ax1,'children');
    fig2 = get(ax2,'children');
    fig3 = get(ax3,'children');

    copyobj(fig1,s1);
    copyobj(fig2,s2);
    copyobj(fig3,s3);

    if h3.Position(3) ~= 800
        h3.Position(2) = 150;
        h3.Position(3) = 800;
        h3.Position(4) = 800;
    end

    subp = get(h3,'children');

    ax1 = subp(1);
    grid on;
    ax2 = subp(2);
    grid on;
    ax3 = subp(3);
    grid on;

    fontsize = 26;
    %%%
    set(ax1, 'fontsize', fontsize); %KL btm
    ax1.GridAlpha = 0.4;

    set(ax1, 'XGrid', 'on');
    set(ax1, 'YGrid', 'on');
    ax1.XTick = [1 2 3];
    ax1.XTickLabel = {'MF-Exact', 'VOGN-1', 'Vadam'};
    ax1.XLim = [0.5 3.5];
    ax1.YLabel.String = 'KL';
    ax1.YLabel.Position = [0.20, 19, -1];

    ax1.YLim = [-1 35];
    ax1.YTick = [0, 10, 20, 30];
    ax1.YTickLabel = {'0', '10', '20', '30'};

    %%%
    set(ax2, 'fontsize', fontsize); % one in middle
    ax2.GridAlpha = 0.4;

    set(ax2,'xticklabel',{[]}) ;
    set(ax2, 'XGrid', 'on');
    set(ax2, 'YGrid', 'on');
    ax2.XTick = [1 2 3];
    ax2.XLim = [0.5 3.5];
    ax2.YLabel.String = 'LogLoss';

    ax2.YLim = [0.115 0.175];
    ax2.YTick = [0.12 0.14 0.16];
    ax2.YTickLabel = {'0.12', '0.14', '0.16'};
    %ax2.YTick = [0.13 0.15 0.17];
    %ax2.YTickLabel = {'0.13', '0.15', '0.17'};

    %%%
    set(ax3, 'fontsize', fontsize); % top
    ax3.GridAlpha = 0.4;

    set(ax3,'xticklabel',{[]}) ;
    set(ax3, 'XGrid', 'on');
    set(ax3, 'YGrid', 'on');
    ax3.XTick = [1 2 3];
    ax3.XLim = [0.5 3.5];
    ax3.YLabel.String = 'Negative ELBO';

    ax3.YLim = [0.235 0.33];
    ax3.YTick = [0.24 0.28 0.32];
    ax3.YTickLabel = {'0.24', '0.28', '0.32'};

    f = h3;
end
