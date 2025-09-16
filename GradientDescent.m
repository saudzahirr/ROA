clear all; clc; format long;

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
%  Newton–Raphson Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        % Initial guess
        V = [1.0; V_range(i)];      % Slack fixed at 1.0
        delta = [0.0; delta_range(j)];

        iter = 0;
        converged = false;

        while iter < max_iter
            iter = iter + 1;

            % Complex voltages
            Vc = V .* exp(1j*delta);

            % Calculate power injections
            I = Y_bus * Vc;
            S_calc = Vc .* conj(I);
            P_calc = real(S_calc);
            Q_calc = imag(S_calc);

            % Mismatch (only PQ bus)
            dP = P_spec(pq_bus_id) - P_calc(pq_bus_id);
            dQ = Q_spec(pq_bus_id) - Q_calc(pq_bus_id);
            mismatch = [dP; dQ];

            if max(abs(mismatch)) < tolerance
                converged = true;
                break;
            end

            % Jacobian (2x2 for 1 PQ bus: dP/dθ, dP/dV, dQ/dθ, dQ/dV)
            k = pq_bus_id; % bus 2
            Vk = V(k); deltak = delta(k);

            Pk = P_calc(k); Qk = Q_calc(k);

            % dP/dθk
            dP_dtheta = -Qk - (B(k,k)*Vk^2);
            % dP/dVk
            dP_dV = (Pk/Vk) + G(k,k)*Vk;
            % dQ/dθk
            dQ_dtheta = Pk - G(k,k)*Vk^2;
            % dQ/dV
            dQ_dV = (Qk/Vk) - B(k,k)*Vk;

            J = [dP_dtheta, dP_dV;
                 dQ_dtheta, dQ_dV];

            % Correction
            alpha = 0.01;
            grad = J.' * mismatch;   % Steepest descent direction
            dx = alpha * grad;        % Step

            % Update θ2, V2
            delta(k) = delta(k) + dx(1);
            V(k)     = V(k) + dx(2);
        end

        if converged
            IterationCount(i,j) = iter;
        else
            IterationCount(i,j) = NaN; % Divergence shown as black
        end
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
if max_iter > 3
    levels = 12:3:25; 
    levels = levels(levels <= max_iter);

    if ~isempty(levels)
        [C,h] = contour(delta_range, V_range, IterPlot, levels, 'k', 'LineWidth', 1.2);

        H = clabel(C,h,'FontSize', 12, 'Color', 'k', 'LabelSpacing', 600, ...
               'BackgroundColor', 'none', 'EdgeColor', 'none');
    end
end

cmap = [0 0 0; jet(max_iter)];
colormap(cmap)
caxis([-1 max_iter])

colorbar
xlabel('\theta_2 [rad]')
ylabel('v_2 [pu]')
if addTitle
    title('Region of Attraction of Gradient Descent Method')
end

% Set font to Times New Roman, size 12
set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf, 'GradientDescent-RegionOfAttraction.png');
