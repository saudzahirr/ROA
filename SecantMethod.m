clear all; clc; format long;

global addTitle;

%% -------------------------------
%  2-BUS SYSTEM - SECANT METHOD
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

% Indices
pq_bus_id = find(bus_type == 0);
slack_id  = find(bus_type == 1);

n_bus = 2;
n_pq  = length(pq_bus_id);

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
max_iter = 25;

%% -------------------------------
%  Secant Method Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        try
            % Initial guesses (need two points for secant method)
            V0 = [1.0; V_range(i)];
            delta0 = [0.0; delta_range(j)];

            % Second point (small perturbation from first)
            perturbation = 0.01;
            V1 = [1.0; V_range(i) + perturbation];
            delta1 = [0.0; delta_range(j) + perturbation];

            k = pq_bus_id; % PQ bus index

            iter = 0;
            converged = false;

            % Calculate initial function values
            [f0, ~] = calculate_mismatch(V0, delta0, Y_bus, P_spec, Q_spec, pq_bus_id);
            [f1, ~] = calculate_mismatch(V1, delta1, Y_bus, P_spec, Q_spec, pq_bus_id);

            while iter < max_iter
                iter = iter + 1;

                % Secant method for each variable
                % x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

                % Check for convergence
                if norm(f1) < tolerance
                    converged = true;
                    break;
                end

                % Avoid division by zero
                df = f1 - f0;
                if norm(df) < 1e-14
                    break;
                end

                % Calculate secant step for θ and V simultaneously
                % Treating as a 2D secant method
                dx_theta = delta1(k) - delta0(k);
                dx_V = V1(k) - V0(k);
                df_theta = f1(1) - f0(1);  % P mismatch difference
                df_V = f1(2) - f0(2);      % Q mismatch difference

                % Secant updates (simplified multi-dimensional approach)
                if abs(df_theta) > 1e-14
                    delta_new_k = delta1(k) - f1(1) * dx_theta / df_theta;
                else
                    delta_new_k = delta1(k);
                end

                if abs(df_V) > 1e-14
                    V_new_k = V1(k) - f1(2) * dx_V / df_V;
                else
                    V_new_k = V1(k);
                end

                % Update points
                V0 = V1;
                delta0 = delta1;
                f0 = f1;

                V1 = [1.0; V_new_k];
                delta1 = [0.0; delta_new_k];

                % Calculate new function value
                [f1, ~] = calculate_mismatch(V1, delta1, Y_bus, P_spec, Q_spec, pq_bus_id);

                % Bound the variables to prevent divergence
                if V1(k) < 0.1 || V1(k) > 3.0 || abs(delta1(k)) > pi
                    break;
                end
            end

            if converged
                IterationCount(i,j) = iter;
            end

        catch
            % Handle numerical issues
            IterationCount(i,j) = NaN;
        end
    end
end

%% -------------------------------
%  Helper Function
% -------------------------------
function [f, J] = calculate_mismatch(V, delta, Y_bus, P_spec, Q_spec, pq_bus_id)
    % Calculate power mismatch
    Vc = V .* exp(1j*delta);
    I = Y_bus * Vc;
    S_calc = Vc .* conj(I);
    P_calc = real(S_calc);
    Q_calc = imag(S_calc);

    % Mismatch (only PQ bus)
    dP = P_spec(pq_bus_id) - P_calc(pq_bus_id);
    dQ = Q_spec(pq_bus_id) - Q_calc(pq_bus_id);
    f = [dP; dQ];

    % Jacobian (optional for secant method, but useful for stability)
    if nargout > 1
        k = pq_bus_id;
        Vk = V(k);
        Pk = P_calc(k); Qk = Q_calc(k);
        G = real(Y_bus); B = imag(Y_bus);

        dP_dtheta = -Qk - (B(k,k)*Vk^2);
        dP_dV = (Pk/Vk) + G(k,k)*Vk;
        dQ_dtheta = Pk - G(k,k)*Vk^2;
        dQ_dV = (Qk/Vk) - B(k,k)*Vk;

        J = [dP_dtheta, dP_dV;
             dQ_dtheta, dQ_dV];
    end
end

%% -------------------------------
%  Plot Region of Attraction
% -------------------------------
figure('Color',[1 1 1])

% Mask divergence as -1 for coloring
IterPlot = IterationCount;
IterPlot(isnan(IterPlot)) = -1;

% Plot smooth background
contourf(delta_range, V_range, IterPlot, 50, 'LineStyle','none');
hold on;

% Overlay iteration contours
valid_data = IterPlot(IterPlot > 0);
if ~isempty(valid_data)
    levels = [5, 10, 15, 20];
    levels = levels(levels <= max(valid_data));

    if ~isempty(levels)
        [C,h] = contour(delta_range, V_range, IterPlot, levels, 'k', 'LineWidth', 1.2);
        clabel(C,h,'FontSize', 12, 'Color', 'k', 'LabelSpacing', 600, ...
               'BackgroundColor', 'none', 'EdgeColor', 'none');
    end
end

cmap = [0 0 0; jet(max_iter)];
colormap(cmap)
if ~isempty(valid_data)
    caxis([-1 max(valid_data)])
else
    caxis([-1 max_iter])
end

colorbar
xlabel('\theta_2 [rad]')
ylabel('v_2 [pu]')
if addTitle
    title('Region of Attraction of Secant Method')
end

% Set font to Times New Roman, size 12
set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf, 'SecantMethod-RegionOfAttraction.png');
