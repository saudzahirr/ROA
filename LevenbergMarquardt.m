clear all; clc; format long;

global addTitle;

%% -------------------------------
%  2-BUS SYSTEM
% -------------------------------
Z12 = 0 + 1j*0.1;
Y12 = -1/Z12;
Y_bus = [ -Y12,   Y12;
           Y12,  -Y12 ];
G = real(Y_bus);
B = imag(Y_bus);

P_spec = [0; -0.9];
Q_spec = [0; -0.6];
bus_type = [1; 0];
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
%  Levenberg–Marquardt Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        V = [1.0; V_range(i)];
        delta = [0.0; delta_range(j)];

        iter = 0;
        converged = false;

        lambda = 0.01;

        while iter < max_iter
            iter = iter + 1;

            [P_calc, Q_calc] = computePQ(V, delta, Y_bus);
            dP = P_spec(pq_bus_id) - P_calc(pq_bus_id);
            dQ = Q_spec(pq_bus_id) - Q_calc(pq_bus_id);
            F = [dP; dQ];

            if max(abs(F)) < tolerance
                converged = true;
                break;
            end

            % Jacobian
            k = pq_bus_id; Vk = V(k);
            Pk = P_calc(k); Qk = Q_calc(k);
            dP_dtheta = -Qk - (B(k,k)*Vk^2);
            dP_dV     = (Pk/Vk) + G(k,k)*Vk;
            dQ_dtheta = Pk - G(k,k)*Vk^2;
            dQ_dV     = (Qk/Vk) - B(k,k)*Vk;
            J = [dP_dtheta, dP_dV; dQ_dtheta, dQ_dV];

            A = J' * J;
            g = J' * F;

            % LM step: (A + lambda*I) dx = g
            I2 = eye(size(A));
            if rcond(A + lambda*I2) < 1e-14
                dx = pinv(A + lambda*I2) * g;
            else
                dx = (A + lambda*I2) \ g;
            end

            % Trial update
            delta_trial = delta; V_trial = V;
            delta_trial(k) = delta_trial(k) + dx(1);
            V_trial(k)     = V_trial(k)     + dx(2);

            [P_calc_t, Q_calc_t] = computePQ(V_trial, delta_trial, Y_bus);
            dP_t = P_spec(pq_bus_id) - P_calc_t(pq_bus_id);
            dQ_t = Q_spec(pq_bus_id) - Q_calc_t(pq_bus_id);
            F_trial = [dP_t; dQ_t];

            if norm(F_trial) < norm(F)
                % accept and decrease lambda
                delta = delta_trial;
                V = V_trial;
                lambda = lambda / 10;
            else
                % reject and increase lambda
                lambda = lambda * 10;
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
if max_iter > 3
    levels = [4, 5, 6, 7, 8, 9]; 
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
    title('Region of Attraction of Levenberg-Marquardt Method')
end

set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
saveas(gcf, 'LevenbergMarquardt-RegionOfAttraction.png');

%% -------------------------------
%  Helper: compute P and Q for each bus
%% -------------------------------
function [P, Q] = computePQ(V, delta, Y_bus)
    Vc = V .* exp(1j*delta);
    I = Y_bus * Vc;
    S_calc = Vc .* conj(I);
    P = real(S_calc);
    Q = imag(S_calc);
end
