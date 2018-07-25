% @Author: aaronmishkin
% @Date:   2018-06-11T23:21:24-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T11:47:18-07:00



function [f] = gen_box_plot(data, name, y_axis, dataset_name, show_axis, show_labels, show_title)

    f = figure('Position', [50,50,700,400]);
    if show_labels
        bx = boxplot(data, 'LabelOrientation', 'horizontal', 'PlotStyle', 'Traditional', 'Notch', 'on', 'Labels', {'E.', 'VOGN-1', 'Vadam'});
    else
        bx = boxplot(data, 'LabelOrientation', 'horizontal', 'PlotStyle', 'Traditional', 'Notch', 'on', 'Labels', {'','', ''});
    end

    if show_title
        title(dataset_name);
    end
    set(bx, {'linew'}, {8})
    grid on;


    % xlabel('L');
    if show_axis
        ylabel(y_axis);
    end
    %set(gca, 'yscale', 'log')
    set(findall(gcf,'-property','FontSize'),'FontSize',38)
    set(findobj(gcf,'-regexp','Tag','\w*Whisker'),'LineStyle','-')
    tightfig();

    hold on;
    h = gcf;

    fontsize = 24;
    ax = gca;
    set(ax, 'fontsize', fontsize);
    %%%

    h.Position(3) = 700;
    h.Position(4) = 250;
    ax.XTickLabel = {'VI-Exact', 'VOGN-1', 'Vadam'};
    ax.GridAlpha = 0.7;

    axesObjs = get(h, 'Children');
    dataObjs = get(axesObjs, 'Children');
    cell_obj = get(dataObjs, 'Children');

    %%%%

    box_plot = findobj('tag','boxplot');
    boxes = get(box_plot, 'Children');  %handles to low-level graphics objects in axes
    if iscell(boxes)
        boxes = boxes{1};
    end

    Vadam = boxes([4, 7]);
    VOGN1 = boxes([5, 8]);
    VI = boxes([6, 9]);

    for i = [1 : 1]
        set(VOGN1(i), 'color', 'b')
        set(Vadam(i), 'color', 'r')
        set(VI(i), 'color', [0.25, 0.25, 0.25])
    end

    alpha=0.8;
    for i = [1 : 1]
        set(VOGN1(i+1), 'color', [0,0,1,alpha])
        set(Vadam(i+1), 'color', [1,0,0,alpha])
        set(VI(i+1), 'color', [0.25, 0.25, 0.25, alpha])
    end

    linewidth = 5;
    for i = [1 : 21]
         set(boxes(i),'linewidth', linewidth)
    end


    set(f,'Units','Inches');
    pos = get(f,'Position');
    set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
end
