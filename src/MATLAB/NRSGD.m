global addTitle;

%% -------------------------------
%  2-BUS SYSTEM
% -------------------------------
% Bus 1: Slack (V1 = 1.0 ∠ 0 pu)
% Bus 2: PQ bus (S2 = 0.9 + j0.6 pu)
% Line: assume Z = j0.1 pu
% -------------------------------

% Y-bus
Z12 = 0 + 1j*0.1;
Y12 = -1/Z12;
Y_bus = [ -Y12,   Y12;
           Y12,  -Y12 ];

G = real(Y_bus);
B = imag(Y_bus);

% Scheduled powers
P_spec = [0; -0.9];   % P2 = -0.9 pu (load)
Q_spec = [0; -0.6];   % Q2 = -0.6 pu (load)

% Bus types: 1 = slack, 0 = PQ
bus_type = [1; 0];
pq_bus_id = find(bus_type == 0);

n_bus = 2;

%% -------------------------------
%  Region of Attraction Grid
% -------------------------------
N = 256;
V_range = linspace(0, 5, N);
delta_range = linspace(-pi/2, pi/2, N);

nV = length(V_range);
nd = length(delta_range);

IterationCount = NaN(nV, nd);

tolerance = 1e-5;
max_iter = 25;   % usually slower than NR

eta = 0.01;  beta1 = 0.9;  beta2 = 0.999;   % Adam step size and decay rates
nr_iters = 10;   % Newton-Raphson iterations per phase
sgd_steps = 50;  % gradient steps per phase

sfun = @(x) (x(2) * exp(1j * x(1))) * conj(Y_bus(2,1) * 1.0 + Y_bus(2,2) * x(2) * exp(1j * x(1)));
misf = @(x) [P_spec(2) - real(sfun(x)); Q_spec(2) - imag(sfun(x))];
jacf = @(x) [-imag(sfun(x)) - B(2,2) * x(2)^2,  real(sfun(x)) / x(2) + G(2,2) * x(2);
              real(sfun(x)) - G(2,2) * x(2)^2,  imag(sfun(x)) / x(2) - B(2,2) * x(2)];

%% -------------------------------
%  Newton-Raphson with stochastic gradient descent phases (re-implementation of Costilla-Enriquez et al.) sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        % state x = [theta2; v2];  mismatch F(x) = specified - calculated
        x = [delta_range(j); V_range(i)];
        iter = 0;                                      % counts Newton-Raphson iterations
        converged = max(abs(misf(x))) < tolerance;
        if converged, iter = 1; end
        mAdam = zeros(2,1);  uAdam = zeros(2,1);  tAdam = 0;

        while ~converged && iter < max_iter
            xStart = x;  ok = true;  nrm = inf;
            for kNR = 1:nr_iters                       % ---- Newton-Raphson phase
                if iter >= max_iter, break; end
                iter = iter + 1;
                J = jacf(x);
                if ~(abs(det(J)) > 0), ok = false; break; end
                x = x + J \ misf(x);
                nrm = max(abs(misf(x)));
                if ~isfinite(nrm) || nrm > 1e8, ok = false; break; end
                if nrm < tolerance
                    converged = true;  iter = iter + 1;
                    break;
                end
            end
            if converged || iter >= max_iter, break; end
            if ~ok || nrm > 1e3, x = xStart; end       % Newton-Raphson diverged: restart the phase
            for kSGD = 1:sgd_steps                     % ---- stochastic gradient phase (Adam)
                g = -(jacf(x).' * misf(x));
                tAdam = tAdam + 1;
                mAdam = beta1 * mAdam + (1 - beta1) * g;
                uAdam = beta2 * uAdam + (1 - beta2) * g.^2;
                x = x - eta * (mAdam / (1 - beta1^tAdam)) ./ (sqrt(uAdam / (1 - beta2^tAdam)) + 1e-8);
            end
        end
        if ~all(isfinite(x)), converged = false; end

        if converged
            IterationCount(i,j) = iter;
        else
            IterationCount(i,j) = NaN;
        end
    end
end

%% -------------------------------
%  Plot Region of Attraction
% -------------------------------
figure('Color',[1 1 1])

IterPlot = IterationCount;
IterPlot(isnan(IterPlot)) = -1;  

contourf(delta_range, V_range, IterPlot, 50, 'LineStyle','none'); 
hold on;

if max_iter > 5
    levels = [5 6 7 8]; 
    levels = levels(levels <= max_iter);

    if ~isempty(levels)
        % Clean iteration contours: the map is smoothed with a 7 x 7 median filter,
        % only long contour lines are drawn (no islands of the irregular bands),
        % and each level is labelled once, in the middle of its longest line.
        hold on;
        Fs = IterPlot;
        Fs(Fs < 0) = max(levels) + 10;                    % not converged: outside every contour
        Fs = movmedian(movmedian(Fs, 7, 1), 7, 2);
        for level = levels
            Cl = contourc(delta_range, V_range, Fs, [level - 0.5, level - 0.5]);
            kc = 1;  best = [];
            while kc < size(Cl, 2)
                nc = Cl(2, kc);
                xy = Cl(:, kc + 1:kc + nc);
                if nc >= 60
                    plot(xy(1, :), xy(2, :), 'k', 'LineWidth', 1.2);
                    if isempty(best) || nc > size(best, 2), best = xy; end
                end
                kc = kc + nc + 1;
            end
            if ~isempty(best)
                mid = round(size(best, 2) / 2);
                text(best(1, mid), best(2, mid), sprintf('%d', level), 'FontSize', 12, 'Color', 'k', ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'BackgroundColor', 'none', 'Margin', 0.1, 'Clipping', 'on');
            end
        end
    end
end

cmap = [0 0 0; jet(max_iter)];
colormap(cmap)
caxis([-1 max_iter])

colorbar
xlabel('\theta_2 [rad]')
ylabel('v_2 [pu]')
if addTitle
    title('Region of Attraction of Newton-Raphson + SGD Method')
end

set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf,'NRSGD-RegionOfAttraction.png');
