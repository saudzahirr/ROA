function k = natureStyle(fig, widthMM, heightMM)
%NATURESTYLE  Resize a figure to its final print size and apply Nature Portfolio lettering.
%   natureStyle(fig, widthMM, heightMM) sets the figure to widthMM x heightMM
%   millimetres (88 mm = one column, 180 mm = two columns), black Arial lettering
%   of 7 pt (6 pt for tick labels) and 0.5 pt line weights, and lets the axes
%   fill the figure, so that the exported file has the lettering size of the
%   printed page. Contour labels are restyled by the caller with CLABEL.
try, fig.Theme = 'light'; catch, end        %#ok<NOCOM> newer releases: fixed light colours
% the printed size is fixed through the paper position, which does not depend on
% the window manager honouring the requested window size
set(fig, 'Units', 'centimeters', 'Position', [2 2 widthMM/10 heightMM/10], 'Color', 'w', ...
    'InvertHardcopy', 'off', 'PaperUnits', 'centimeters', 'PaperPositionMode', 'manual', ...
    'PaperSize', [widthMM/10 heightMM/10], 'PaperPosition', [0 0 widthMM/10 heightMM/10]);
drawnow; pause(0.3);
% The window may not get the requested size (for example in batch mode). The figure is
% scaled to the paper size when printed, so all point sizes are multiplied by
% k = actual window width / requested width to come out right on paper.
pos = get(fig, 'Position');
k = pos(3) / (widthMM/10);
ax = findall(fig, 'Type', 'axes');
set(ax, 'FontName', 'Arial', 'FontSize', 6*k, 'LineWidth', 0.5*k, 'TickDir', 'out', ...
    'TickLength', [0.012 0.012], 'Layer', 'top', 'Box', 'on', 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
for q = 1:numel(ax)
    set([get(ax(q), 'XLabel'), get(ax(q), 'YLabel')], 'FontSize', 7*k, 'FontName', 'Arial', 'Color', 'k');
    % margins in millimetres: left, bottom, right (room for the colour bar), top
    m = [10 8.5 15 2];
    set(ax(q), 'Units', 'normalized', 'Position', ...
        [m(1)/widthMM, m(2)/heightMM, 1 - (m(1) + m(3))/widthMM, 1 - (m(2) + m(4))/heightMM]);
end
cb = findall(fig, 'Type', 'ColorBar');
for q = 1:numel(cb)
    set(cb(q), 'FontName', 'Arial', 'FontSize', 6*k, 'LineWidth', 0.5*k, 'Color', 'k');
    cb(q).Label.String = 'Iterations to converge';
    cb(q).Label.FontSize = 7*k;
    cb(q).Label.FontName = 'Arial';
    cb(q).Label.Color = 'k';
end
c = findall(fig, 'Type', 'contour');
for q = 1:numel(c)
    if ~strcmp(c(q).LineStyle, 'none')
        c(q).LineWidth = 0.4*k;               % labelled iteration contours
    end
end
drawnow;
end
