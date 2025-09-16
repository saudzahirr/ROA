clear all; clc; format long;

global addTitle;

%% -------------------------------
%  2-BUS SYSTEM - BROYDEN'S METHOD
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
%  Broyden's Method Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        try
            % Initial guess
            V = [1.0; V_range(i)];
            delta = [0.0; delta_range(j)];

            k = pq_bus_id; % PQ bus index

            % Calculate initial mismatch and Jacobian
            [f, J] = calculate_mismatch(V, delta, Y_bus, P_spec, Q_spec, pq_bus_id, G, B);

            % Initialize Broyden matrix (start with Jacobian)
            B_broyden = J;

            iter = 0;
            converged = false;

            while iter < max_iter
                iter = iter + 1;

                % Check for convergence
                if max(abs(f)) < tolerance
                    converged = true;
                    break;
                end

                % Solve for step
                if abs(det(B_broyden)) > 1e-12
                    dx = B_broyden \ (-f);
                else
                    break; % Singular matrix
                end

                % Update variables
                x_old = [delta(k); V(k)];
                x_new = x_old + dx;

                % Bound the variables
                if x_new(2) < 0.1 || x_new(2) > 3.0 || abs(x_new(1)) > pi
                    break;
                end

                delta(k) = x_new(1);
                V(k) = x_new(2);

                % Calculate new mismatch
                f_old = f;
                [f, ~] = calculate_mismatch(V, delta, Y_bus, P_spec, Q_spec, pq_bus_id, G, B);

                % Broyden update
                % B_{k+1} = B_k + ((y - B_k*s) * s^T) / (s^T * s)
                % where y = f_{k+1} - f_k, s = x_{k+1} - x_k

                y = f - f_old;
                s = dx;

                if norm(s) > 1e-12
                    B_broyden = B_broyden + (y - B_broyden*s) * s' / (s'*s);
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
function [f, J] = calculate_mismatch(V, delta, Y_bus, P_spec, Q_spec, pq_bus_id, G, B)
    % Calculate power mismatch and Jacobian
    Vc = V .* exp(1j*delta);
    I = Y_bus * Vc;
    S_calc = Vc .* conj(I);
    P_calc = real(S_calc);
    Q_calc = imag(S_calc);

    % Mismatch (only PQ bus)
    dP = P_spec(pq_bus_id) - P_calc(pq_bus_id);
    dQ = Q_spec(pq_bus_id) - Q_calc(pq_bus_id);
    f = [dP; dQ];

    % Jacobian
    if nargout > 1
        k = pq_bus_id;
        Vk = V(k);
        Pk = P_calc(k); Qk = Q_calc(k);

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
    levels = [1:2:15];
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
    title('Region of Attraction of Broyden''s Method')
end

% Set font to Times New Roman, size 12
set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf, 'Broyden-RegionOfAttraction.png');
