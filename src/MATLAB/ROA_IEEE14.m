%% Region of attraction on the IEEE 14-bus system
% The starting voltage magnitude and angle of one load bus are swept over the grid
% of the two-bus maps (v in [0, 5] p.u., theta in [-pi/2, pi/2]); all other buses
% start from the flat start.  Two methods are run from every starting point:
%   Newton-Raphson, and
%   the guarded Newton-Landweber method (voltage guard, angle cap, look-ahead trial,
%   accelerated Landweber step, restoration step).
% A run counts if it reaches the operating point (all bus voltages within 1e-4 p.u.
% of the reference solution) within 25 iterations (tolerance 1e-5 p.u.).
%
% The script is self-contained: the bus admittance matrix, the specified injections,
% the flat start and the reference solution of the IEEE 14-bus system are embedded.
% Maps are written to ../../figures/roa (or the folder named by the environment variable ROA_FIG_DIR) as IEEE14_bus<k>_<NR|Guarded>_88mm.png.
%
%   matlab -batch "ROA_IEEE14"

set(groot, 'DefaultFigureVisible', 'off');     % no figure windows

%% -------------------------------
%  IEEE 14-BUS SYSTEM (per unit, 100 MVA base; buses numbered 1..14)
% -------------------------------
Ybus = [6.02502905576822-19.4470702055144i -4.99913160079803+15.2630865231796i 0+0i 0+0i -1.02589745497019+4.23498368233483i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i;
        -4.99913160079803+15.2630865231796i 9.52132361081478-30.2721153987791i -1.1350191923074+4.78186315175772i -1.68603315061494+5.11583832587208i -1.7011396670944+5.19392739796971i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i;
        0+0i -1.1350191923074+4.78186315175772i 3.12099490223296-9.82238012935164i -1.98597570992556+5.06881697759392i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i;
        0+0i -1.68603315061494+5.11583832587208i -1.98597570992556+5.06881697759392i 10.5129895220362-38.6541712076078i -6.84098066149567+21.5785539816916i 0+0i 0+4.88951266031734i 0+0i 0+1.8554995578159i 0+0i 0+0i 0+0i 0+0i 0+0i;
        -1.02589745497019+4.23498368233483i -1.7011396670944+5.19392739796971i 0+0i -6.84098066149567+21.5785539816916i 9.56801778356026-35.5336394560448i 0+4.25744533525338i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i;
        0+0i 0+0i 0+0i 0+0i 0+4.25744533525338i 6.57992340746622-17.3407328099191i 0+0i 0+0i 0+0i 0+0i -1.95502856317726+4.09407434424044i -1.52596744045097+3.1759639650294i -3.09892740383799+6.10275544819312i 0+0i;
        0+0i 0+0i 0+0i 0+4.88951266031734i 0+0i 0+0i 0-19.5490059482647i 0+5.67697984672154i 0+9.09008271975275i 0+0i 0+0i 0+0i 0+0i 0+0i;
        0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+5.67697984672154i 0-5.67697984672154i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i;
        0+0i 0+0i 0+0i 0+1.8554995578159i 0+0i 0+0i 0+9.09008271975275i 0+0i 5.32605503946736-24.0925063752679i -3.90204955244743+10.3653941270609i 0+0i 0+0i 0+0i -1.42400548701993+3.0290504569306i;
        0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i -3.90204955244743+10.3653941270609i 5.78293430614783-14.7683378765214i -1.8808847537004+4.40294374946052i 0+0i 0+0i 0+0i;
        0+0i 0+0i 0+0i 0+0i 0+0i -1.95502856317726+4.09407434424044i 0+0i 0+0i 0+0i -1.8808847537004+4.40294374946052i 3.83591331687766-8.49701809370096i 0+0i 0+0i 0+0i;
        0+0i 0+0i 0+0i 0+0i 0+0i -1.52596744045097+3.1759639650294i 0+0i 0+0i 0+0i 0+0i 0+0i 4.01499202727289-5.42793859120161i -2.48902458682192+2.25197462617221i 0+0i;
        0+0i 0+0i 0+0i 0+0i 0+0i -3.09892740383799+6.10275544819312i 0+0i 0+0i 0+0i 0+0i 0+0i -2.48902458682192+2.25197462617221i 6.72494614846623-10.6696935494707i -1.13699415780633+2.31496347510535i;
        0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i 0+0i -1.42400548701993+3.0290504569306i 0+0i 0+0i 0+0i -1.13699415780633+2.31496347510535i 2.56099964482626-5.34401393203596i];
Sbus = [2.324-0.169i; 0.183+0.297i; -0.942+0.044i; -0.478+0.039i; -0.076-0.016i; -0.112+0.047i; -0+0i; 0+0.174i; -0.295-0.166i; -0.09-0.058i; -0.035-0.018i; -0.061-0.016i; -0.135-0.058i; -0.149-0.05i];                                  % specified complex injections
Vflat = [1.06+0i; 1.045+0i; 1.01+0i; 1+0i; 1+0i; 1.07+0i; 1+0i; 1.09+0i; 1+0i; 1+0i; 1+0i; 1+0i; 1+0i; 1+0i];                                % flat start (set-points at generators)
Vref = [1.06+0i; 1.0410510878564-0.0907614040934373i; 0.985192522280656-0.222476277482069i; 1.001230129403-0.182187251037279i; 1.00758360184552-0.155511400424921i; 1.03721049678572-0.262858108791664i; 1.03279368287084-0.245277244129254i; 1.06050343857989-0.251858009124656i; 1.02024437993435-0.272200668377919i; 1.01470984678035-0.273738212235293i; 1.02188611774487-0.269815031634412i; 1.01887272102613-0.274447230358386i; 1.01384554676904-0.274625111057152i; 0.995247493185494-0.286015202518851i];                                  % operating point
pv = [2 3 6 8];                                      % generator buses
pq = [4 5 7 9 10 11 12 13 14];                                      % load buses
pvpq = [pv pq];
nA = numel(pvpq);                               % number of angle unknowns

%% -------------------------------
%  Settings
% -------------------------------
N = 96;                                        % grid points per axis
V_range = linspace(0, 5, N);
delta_range = linspace(-pi/2, pi/2, N);
sweepBuses = [4 5];                             % load buses whose starting voltage is swept
tolerance = 1e-5;
max_iter = 25;
V_guard = 0.5;  dtheta_cap = 0.5;  restore = 0.5;  m_trial = 2;  rho = 0.25;

outDir = getenv('ROA_FIG_DIR');             % set ROA_FIG_DIR to export elsewhere
if isempty(outDir); outDir = fullfile('..', '..', 'figures', 'roa'); end
if ~exist(outDir, 'dir'); mkdir(outDir); end
xFlat = [angle(Vflat(pvpq)); abs(Vflat(pq))];
warnState = warning('off', 'all');              % singular Jacobians are handled explicitly

for bus = sweepBuses
    for iMethod = 1:2
        IterationCount = NaN(N, N);
        OtherSolution = false(N, N);            % converged, but not to the operating point
        for i = 1:N
            for j = 1:N
                V0 = Vflat;
                V0(bus) = max(V_range(i), 1e-6) * exp(1j * delta_range(j));
                if iMethod == 1
                    [it, V] = solveNR(V0, Ybus, Sbus, pvpq, pq, tolerance, max_iter);
                else
                    [it, V] = solveGuarded(V0, Ybus, Sbus, pvpq, pq, nA, xFlat, tolerance, max_iter, ...
                                           V_guard, dtheta_cap, restore, m_trial, rho);
                end
                if ~isnan(it)
                    if max(abs(V - Vref)) < 1e-4
                        IterationCount(i, j) = it;
                    else
                        OtherSolution(i, j) = true;
                    end
                end
            end
        end
        names = {'NR', 'Guarded'};
        reached = ~isnan(IterationCount);
        fprintf('bus %d, %s: operating point from %.1f%% of the grid (other solution %.1f%%), mean %.2f iterations, max %d\n', ...
            bus, names{iMethod}, 100 * mean(reached(:)), 100 * mean(OtherSolution(:)), ...
            mean(IterationCount(reached)), max(IterationCount(reached)));
        drawMap(delta_range, V_range, IterationCount, OtherSolution, max_iter, bus, ...
                fullfile(outDir, sprintf('IEEE14_bus%d_%s_88mm.png', bus, names{iMethod})));
    end
end
warning(warnState);
set(groot, 'DefaultFigureVisible', 'on');


%% -------------------------------
%  Power flow equations
% -------------------------------
function F = mismatch(V, Ybus, Sbus, pvpq, pq)
% specified minus calculated: active power at PV and PQ buses, reactive power at PQ buses
dS = Sbus - V .* conj(Ybus * V);
F = [real(dS(pvpq)); imag(dS(pq))];
end

function J = jacobian(V, Ybus, pvpq, pq)
Ibus = Ybus * V;
dVa = 1j * diag(V) * conj(diag(Ibus) - Ybus * diag(V));
dVm = diag(V) * conj(Ybus * diag(V ./ abs(V))) + conj(diag(Ibus)) * diag(V ./ abs(V));
J = [real(dVa(pvpq, pvpq)), real(dVm(pvpq, pq)); imag(dVa(pq, pvpq)), imag(dVm(pq, pq))];
end

function V = unpack(x, V, pvpq, pq)
nA = numel(pvpq);
Va = angle(V);  Vm = abs(V);
Va(pvpq) = x(1:nA);
Vm(pq) = x(nA + 1:end);
V = Vm .* exp(1j * Va);
end

%% -------------------------------
%  Newton-Raphson
% -------------------------------
function [it, V] = solveNR(V, Ybus, Sbus, pvpq, pq, tol, maxIter)
x = [angle(V(pvpq)); abs(V(pq))];
F = mismatch(V, Ybus, Sbus, pvpq, pq);
it = 0;
converged = max(abs(F)) < tol;
while ~converged && it < maxIter
    it = it + 1;
    dx = jacobian(V, Ybus, pvpq, pq) \ F;
    if ~all(isfinite(dx)), break; end
    x = x + dx;
    V = unpack(x, V, pvpq, pq);
    F = mismatch(V, Ybus, Sbus, pvpq, pq);
    nrm = max(abs(F));
    if ~isfinite(nrm) || nrm > 1e12, break; end
    converged = nrm < tol;
end
if ~converged, it = NaN; end
end

%% -------------------------------
%  Guarded Newton-Landweber method
% -------------------------------
function [it, V] = solveGuarded(V, Ybus, Sbus, pvpq, pq, nA, xFlat, tol, maxIter, Vg, cap, beta, mTrial, rho)
x = [angle(V(pvpq)); abs(V(pq))];
F = mismatch(V, Ybus, Sbus, pvpq, pq);
normAt = @(xt) norm(mismatch(unpack(xt, V, pvpq, pq), Ybus, Sbus, pvpq, pq));
it = 0;
converged = max(abs(F)) < tol;
trial = false;  fBar = inf;  xNext = [];  left = 0;      % look-ahead trial
phi = inf;                                               % mismatch of the last failed trial
xPrev = [];  jAcc = 0;                                   % accelerated Landweber step
while ~converged && it < maxIter
    it = it + 1;
    J = jacobian(V, Ybus, pvpq, pq);
    f0 = norm(F);
    dx = J \ F;
    exists = all(isfinite(dx));
    admissible = exists && min(x(nA + 1:end) + dx(nA + 1:end)) >= Vg;       % voltage guard on the full step
    if admissible && max(abs(dx(1:nA))) > cap                               % angle cap
        dx = dx * (cap / max(abs(dx(1:nA))));
    end
    if trial
        if admissible, fTrial = normAt(x + dx); else, fTrial = inf; end
        if fTrial < fBar
            x = x + dx;  trial = false;  xPrev = [];  jAcc = 0;
        elseif left > 1 && isfinite(fTrial)
            x = x + dx;  left = left - 1;
        else
            x = xNext;  phi = fBar;  trial = false;                         % trial failed
        end
    elseif admissible && normAt(x + dx) < f0
        x = x + dx;  xPrev = [];  jAcc = 0;                                 % accepted Newton step
    elseif admissible && f0 <= rho * phi
        [xNext, xPrev, jAcc] = fallback(x, xPrev, jAcc, J, F, f0, V);       % start a look-ahead trial
        fBar = f0;  left = mTrial;  trial = true;
        x = x + dx;
    else
        [x, xPrev, jAcc] = fallback(x, xPrev, jAcc, J, F, f0, V);
    end
    V = unpack(x, V, pvpq, pq);
    F = mismatch(V, Ybus, Sbus, pvpq, pq);
    nrm = max(abs(F));
    if ~isfinite(nrm) || nrm > 1e12, break; end
    converged = nrm < tol;
end
if ~converged, it = NaN; end

    function [cand, xPrevOut, jOut] = fallback(xc, xPrevIn, jIn, Jc, Fc, f0c, Vc)
        % accelerated Landweber step from xc, or the restoration step if it would break the guard
        jOut = jIn + 1;
        if isempty(xPrevIn)
            z = xc;
        else
            z = xc + (jOut - 1) / (jOut + 2) * (xc - xPrevIn);
        end
        Vz = unpack(z, Vc, pvpq, pq);
        Jz = jacobian(Vz, Ybus, pvpq, pq);
        cand = z + (Jz.' * mismatch(Vz, Ybus, Sbus, pvpq, pq)) / (norm(Jz, 2)^2 + eps);
        if ~isempty(xPrevIn) && ~(normAt(cand) < f0c)                       % restart the momentum
            jOut = 1;
            cand = xc + (Jc.' * Fc) / (norm(Jc, 2)^2 + eps);
        end
        if ~(min(cand(nA + 1:end)) >= Vg)                                   % restoration step
            cand = xc + beta * (xFlat - xc);
            xPrevOut = [];  jOut = 0;
        else
            xPrevOut = xc;
        end
    end
end

%% -------------------------------
%  Map in the style of the two-bus maps
% -------------------------------
function drawMap(delta_range, V_range, IterationCount, OtherSolution, max_iter, bus, fileName)
% colours: iterations to the operating point (jet), black = no convergence, grey = other solution
IterPlot = IterationCount;
IterPlot(isnan(IterPlot)) = -1;
IterPlot(OtherSolution) = -2;
fig = figure;
imagesc(delta_range, V_range, IterPlot);        % one cell per starting point: no blending of the three classes
set(gca, 'YDir', 'normal');
colormap([0.62 0.62 0.62; 0 0 0; 0 0 0; jet(max_iter)]);    % -2 grey, -1 and 0 black, 1..max_iter jet
caxis([-2.5 max_iter + 0.5]);
cb = colorbar;
cb.Limits = [-1 max_iter];                      % the colour bar shows the iteration scale, as on the two-bus maps
xlabel(sprintf('\\theta_{%d} [rad]', bus));
ylabel(sprintf('v_{%d} [pu]', bus));
k = printStyle(fig, 88, 60);
ax = get(fig, 'CurrentAxes');
set(ax, 'Position', [10/88, 8.5/60, 1 - (10 + 20)/88, 1 - (8.5 + 2)/60], 'XTick', -1.5:0.5:1.5, 'YTick', 0:5);   % room for the colour-bar label
% clean iteration contours, as on the two-bus maps (see export_figures.m)
F = IterationCount;  levels = [5 6 7 8];
F(isnan(F)) = max(levels) + 10;
F = movmedian(movmedian(F, 5, 1), 5, 2);
hold(ax, 'on');
for level = levels
    C = contourc(delta_range, V_range, F, [level - 0.5, level - 0.5]);
    q = 1;  best = [];
    while q < size(C, 2)
        n = C(2, q);
        xy = C(:, q + 1:q + n);
        if n >= 40
            plot(ax, xy(1, :), xy(2, :), 'k', 'LineWidth', 0.4 * k);
            if isempty(best) || n > size(best, 2), best = xy; end
        end
        q = q + n + 1;
    end
    if ~isempty(best)
        mid = round(size(best, 2) / 2);
        text(ax, best(1, mid), best(2, mid), sprintf('%d', level), 'FontName', 'Arial', 'FontSize', 5 * k, ...
            'Color', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Clipping', 'on');
    end
end
axis(ax, [min(delta_range) max(delta_range) min(V_range) max(V_range)]);
print(fig, fileName, '-dpng', '-r600');
close(fig);
end
