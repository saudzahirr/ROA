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

V_guard = 0.5;   % voltage guard (p.u.)
m_trial = 2;     % further Newton steps allowed in a look-ahead trial
dtheta_cap = 0.5;  % largest angle change of a Newton step (rad); longer steps are shortened
restore = 0.5;   % fraction of the way to the flat start taken by a restoration step
rho     = 0.25;  % a new trial needs the mismatch to fall below rho times that of the last failed trial

sfun = @(x) (x(2) * exp(1j * x(1))) * conj(Y_bus(2,1) * 1.0 + Y_bus(2,2) * x(2) * exp(1j * x(1)));
misf = @(x) [P_spec(2) - real(sfun(x)); Q_spec(2) - imag(sfun(x))];
jacf = @(x) [-imag(sfun(x)) - B(2,2) * x(2)^2,  real(sfun(x)) / x(2) + G(2,2) * x(2);
              real(sfun(x)) - G(2,2) * x(2)^2,  imag(sfun(x)) / x(2) - B(2,2) * x(2)];

%% -------------------------------
%  Guarded Newton-Landweber sweep (voltage guard, look-ahead trial, accelerated Landweber step)
% -------------------------------
for i = 1:nV
    for j = 1:nd
        % state x = [theta2; v2];  mismatch F(x) = specified - calculated
        x = [delta_range(j); V_range(i)];
        iter = 0;
        converged = false;
        trial = false;  phi = inf;  xPrev = [];  jAcc = 0;

        while iter < max_iter
            iter = iter + 1;
            F = misf(x);
            if max(abs(F)) < tolerance
                converged = true;
                break;
            end
            f0 = norm(F, 2);
            J = jacf(x);

            % Newton step and voltage guard
            dx = J \ F;
            admissible = all(isfinite(dx)) && rcond(J) > 1e-12 && (x(2) + dx(2) >= V_guard);
            if admissible && abs(dx(1)) > dtheta_cap      % angle cap: shorten an admissible step
                dx = dx * (dtheta_cap / abs(dx(1)));
            end
            if admissible
                fNewton = norm(misf(x + dx), 2);
            else
                fNewton = inf;
            end

            takeLandweber = false;
            if trial                                   % inside a look-ahead trial
                if admissible && fNewton < fBar
                    x = x + dx;  trial = false;  xPrev = [];  jAcc = 0;
                elseif admissible && count < m_trial && isfinite(fNewton)
                    x = x + dx;  count = count + 1;
                else                                   % trial failed: back to the stored iterate
                    x = xBarLW;  phi = fBar;  trial = false;
                end
            elseif admissible && fNewton < f0          % ordinary accepted Newton step
                x = x + dx;  xPrev = [];  jAcc = 0;
            elseif admissible && f0 <= rho * phi       % start a look-ahead trial
                takeLandweber = true;  startTrial = true;
            else
                takeLandweber = true;  startTrial = false;
            end

            if takeLandweber                           % accelerated Landweber step from x
                jAcc = jAcc + 1;
                if isempty(xPrev)
                    z = x;
                else
                    z = x + (jAcc - 1) / (jAcc + 2) * (x - xPrev);
                end
                Jz = jacf(z);
                cand = z + (1 / (norm(Jz, 2)^2 + eps)) * (Jz.' * misf(z));
                if ~isempty(xPrev) && ~(norm(misf(cand), 2) < f0)   % restart the momentum
                    jAcc = 1;
                    cand = x + (1 / (norm(J, 2)^2 + eps)) * (J.' * F);
                end
                if ~(cand(2) >= V_guard)                 % restoration step: the Landweber step would also
                    cand = x + restore * ([0; 1] - x);   % collapse the voltage, so move back towards the flat start
                    xPrev = [];  jAcc = 0;
                else
                    xPrev = x;
                end
                if startTrial
                    xBarLW = cand;  fBar = f0;  count = 1;  trial = true;
                    x = x + dx;                        % first step of the trial
                else
                    x = cand;
                end
            end
        end

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
    title('Region of Attraction of Guarded Newton-Landweber Method')
end

set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf,'Guarded-RegionOfAttraction.png');
