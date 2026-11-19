"""Write ../MATLAB/ROA_IEEE14.m: a self-contained MATLAB script (network data embedded) that
computes the IEEE 14-bus region-of-attraction maps of Newton-Raphson and of the
guarded Newton-Landweber method and exports them in the style of the two-bus maps.

    uv run python scripts/make_matlab_ieee14.py
"""
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.largescale import reference
from roapf.network import build_network

root = Path(__file__).resolve().parents[1]
net = build_network(load_case("case14", str(root / "cases" / "matpower")))
v_ref = reference(net, 1e-8)
Y = net.ybus.toarray()


def cvec(z):
    return "[" + "; ".join(f"{a.real:.15g}{a.imag:+.15g}i" for a in z) + "]"


def cmat(M):
    return "[" + ";\n        ".join(" ".join(f"{a.real:.15g}{a.imag:+.15g}i" for a in row) for row in M) + "]"


def ivec(v):
    return "[" + " ".join(str(int(a) + 1) for a in v) + "]"


TEMPLATE = r"""%% Region of attraction on the IEEE 14-bus system
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
% Maps are written to ../../figures/roa (or to the folder named by ROA_FIG_DIR) as IEEE14_bus<k>_<NR|Guarded>_88mm.png.
%
%   matlab -batch "ROA_IEEE14"

set(groot, 'DefaultFigureVisible', 'off');     % no figure windows

%% -------------------------------
%  IEEE 14-BUS SYSTEM (per unit, 100 MVA base; buses numbered 1..14)
% -------------------------------
Ybus = @YBUS@;
Sbus = @SBUS@;                                  % specified complex injections
Vflat = @VFLAT@;                                % flat start (set-points at generators)
Vref = @VREF@;                                  % operating point
pv = @PV@;                                      % generator buses
pq = @PQ@;                                      % load buses
pvpq = [pv pq];
nA = numel(pvpq);                               % number of angle unknowns

%% -------------------------------
%  Settings
% -------------------------------
N = @N@;                                        % grid points per axis
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
"""

text = (TEMPLATE.replace("@YBUS@", cmat(Y)).replace("@SBUS@", cvec(net.sbus)).replace("@VFLAT@", cvec(net.v_flat))
        .replace("@VREF@", cvec(v_ref)).replace("@PV@", ivec(net.pv)).replace("@PQ@", ivec(net.pq)).replace("@N@", "96"))
out = root.parent / "MATLAB" / "ROA_IEEE14.m"
out.write_text(text, encoding="utf-8", newline="\n")
print(out, "written;  buses in file order:", [int(b) for b in net.case.bus[:, 0]])
