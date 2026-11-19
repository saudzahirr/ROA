%% Region-of-attraction figures at final print size
% Runs the method scripts of this repository unchanged, then resizes the figure
% each one draws and re-exports it with Arial lettering:
%   <name>_88mm.png  one-column panel (88 mm wide)
%   <name>_58mm.png  panel for a three-across figure (58 mm wide)
% Output folder: ../../figures/roa (or the folder named by the environment variable ROA_FIG_DIR) (600 dpi).
%
% So that all maps look alike, the iteration contours drawn by the individual
% scripts are replaced by one common drawing (uniformContours below): the map is
% smoothed with a median filter, only long contour lines are kept and every level
% is labelled once.  The computed iteration counts are not changed.
%
%   matlab -batch "export_figures"
%   setenv('ROA_ONLY','NewtonRaphson,Guarded'); export_figures     % a subset

set(groot, 'DefaultFigureVisible', 'off');   % draw off-screen: no figure windows are opened
global addTitle;
addTitle = false;                       % titles are given in the figure legend

outDir = getenv('ROA_FIG_DIR');             % set ROA_FIG_DIR to export elsewhere
    if isempty(outDir); outDir = fullfile('..', '..', 'figures', 'roa'); end
if ~exist(outDir, 'dir'); mkdir(outDir); end

methodScripts = {'NewtonRaphson', 'LandweberIteration', 'SGD', 'NRSGD', 'Hybrid', 'Guarded', ...
                 'LevenbergMarquardt', 'Iwamoto', 'GaussSeidel', 'SOR', 'FixedPoint', 'FDPF', ...
                 'SecantMethod', 'Broyden', 'GradientDescent', 'Jacobi'};
if ~isempty(getenv('ROA_ONLY')); methodScripts = strsplit(getenv('ROA_ONLY'), ','); end   % optional subset

for iScript = 1:numel(methodScripts)
    scriptName = methodScripts{iScript};
    close all;
    tStart = tic;
    run([scriptName '.m']);             % the published script: computes the sweep and draws the map
    figHandle = findall(groot, 'Type', 'figure');   % the one (invisible) figure drawn by the script
    figHandle = figHandle(1);
    axHandle = get(figHandle, 'CurrentAxes');
    % remove the contour lines and labels drawn by the script (the filled map stays)
    cOld = findall(axHandle, 'Type', 'contour');
    for q = 1:numel(cOld)
        if strcmp(cOld(q).Fill, 'off'), delete(cOld(q)); end
    end
    kids = get(axHandle, 'Children');
    delete(findobj(kids, 'flat', 'Type', 'text'));
    delete(findobj(kids, 'flat', 'Type', 'line'));
    switch scriptName                   % labelled iteration levels
        case 'LandweberIteration',                levels = [11 13 15 16];
        case {'SGD', 'GradientDescent'},          levels = [13 16 19 22];
        otherwise,                                levels = [5 6 7 8];
    end
    sizes = {88, 60, '_88mm.png'; 58, 44, '_58mm.png'};
    for iSize = 1:2
        kScale = printStyle(figHandle, sizes{iSize, 1}, sizes{iSize, 2});
        uniformContours(axHandle, delta_range, V_range, IterationCount, levels, 5*kScale, 0.4*kScale);
        axis(axHandle, [min(delta_range) max(delta_range) min(V_range) max(V_range)]);   % full sweep range
        print(figHandle, fullfile(outDir, [scriptName sizes{iSize, 3}]), '-dpng', '-r600');
    end
    conv = ~isnan(IterationCount);
    fprintf('%s: converged %.1f%%, mean %.2f iterations, done in %.0f s\n', scriptName, ...
        100 * mean(conv(:)), mean(IterationCount(conv)), toc(tStart));
    % the method scripts use the workspace; restore what this loop needs
    methodScripts = {'NewtonRaphson', 'LandweberIteration', 'SGD', 'NRSGD', 'Hybrid', 'Guarded', ...
                     'LevenbergMarquardt', 'Iwamoto', 'GaussSeidel', 'SOR', 'FixedPoint', 'FDPF', ...
                     'SecantMethod', 'Broyden', 'GradientDescent', 'Jacobi'};
    if ~isempty(getenv('ROA_ONLY')); methodScripts = strsplit(getenv('ROA_ONLY'), ','); end
    outDir = getenv('ROA_FIG_DIR');             % set ROA_FIG_DIR to export elsewhere
    if isempty(outDir); outDir = fullfile('..', '..', 'figures', 'roa'); end
    addTitle = false;
end
close all;
set(groot, 'DefaultFigureVisible', 'on');


function uniformContours(ax, x, y, IterationCount, levels, fontSize, lineWidth)
%UNIFORMCONTOURS  The same clean iteration contours on every map.
delete(findall(ax, 'Tag', 'roaUniform'));
F = IterationCount;
F(isnan(F)) = max(levels) + 10;                    % not converged: outside every contour
F = movmedian(movmedian(F, 7, 1), 7, 2);
hold(ax, 'on');
for level = levels
    C = contourc(x, y, F, [level - 0.5, level - 0.5]);
    k = 1;  best = [];
    while k < size(C, 2)
        n = C(2, k);
        xy = C(:, k + 1:k + n);
        if n >= 60
            plot(ax, xy(1, :), xy(2, :), 'k', 'LineWidth', lineWidth, 'Tag', 'roaUniform');
            if isempty(best) || n > size(best, 2), best = xy; end
        end
        k = k + n + 1;
    end
    if ~isempty(best)
        mid = round(size(best, 2) / 2);
        text(ax, best(1, mid), best(2, mid), sprintf('%d', level), 'FontName', 'Arial', 'FontSize', fontSize, ...
            'Color', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'BackgroundColor', 'none', 'Margin', 0.1, 'Clipping', 'on', 'Tag', 'roaUniform');
    end
end
end
