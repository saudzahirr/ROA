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

% Scheduled powers
P_spec = [0; -0.9];
Q_spec = [0; -0.6];

bus_type = [1; 0];
pq_bus_id = find(bus_type == 0);
k = pq_bus_id;

%% -------------------------------
%  Region of Attraction Grid
% -------------------------------
N = 256;
V_range = linspace(0.1, 3, N);
delta_range = linspace(-pi/2, pi/2, N);

IterationCount = NaN(N, N);
tolerance = 1e-6;
max_iter = 25;

%% -------------------------------
%  Iwamoto / Overbye Optimal Multiplier Sweep
% -------------------------------
for i = 1:N
    for j = 1:N
        % Initial guess
        V = [1.0; V_range(i)];
        delta = [0.0; delta_range(j)];
        iter = 0;
        converged = false;

        while iter < max_iter
            iter = iter + 1;

            % Complex voltages
            Vc = V .* exp(1j * delta);
            I = Y_bus * Vc;
            S_calc = Vc .* conj(I);
            P_calc = real(S_calc);
            Q_calc = imag(S_calc);

            % Power mismatch
            dP = P_spec(k) - P_calc(k);
            dQ = Q_spec(k) - Q_calc(k);
            mismatch = [dP; dQ];

            if max(abs(mismatch)) < tolerance
                converged = true;
                break;
            end

            Vk = V(k);
            Pk = P_calc(k);
            Qk = Q_calc(k);

            % Jacobian
            dP_dtheta = -Qk - B(k,k)*Vk^2;
            dP_dV     = (Pk/Vk) + G(k,k)*Vk;
            dQ_dtheta =  Pk - G(k,k)*Vk^2;
            dQ_dV     = (Qk/Vk) - B(k,k)*Vk;
            J = [dP_dtheta, dP_dV;
                 dQ_dtheta, dQ_dV];

            % Solve for Newton step
            if cond(J) > 1e8
                break; % Jacobian singular -> diverged
            end
            dx = J \ mismatch;

            % Compute gradient and curvature (analytical approximation)
            g = J * dx;

            % approximate quadratic term numerically with damping
            epsFD = 1e-4;
            delta_p = delta; V_p = V;
            delta_p(k) = delta_p(k) + epsFD * dx(1);
            V_p(k)     = V_p(k)     + epsFD * dx(2);
            Vc_p = V_p .* exp(1j*delta_p);
            I_p = Y_bus * Vc_p;
            S_p = Vc_p .* conj(I_p);
            f_p = [real(S_p(k)); imag(S_p(k))];

            Vc = V .* exp(1j*delta);
            I0 = Y_bus * Vc;
            S0 = Vc .* conj(I0);
            f0 = [real(S0(k)); imag(S0(k))];

            q = (f_p - f0 - J*epsFD*dx) / (0.5*epsFD^2 + eps);
            denom = (g' * g) + (mismatch' * q);
            numer = -(mismatch' * g);
            p = numer / denom;

            disp([p, numer, denom]);
            % Clamp p for stability
            if ~isfinite(p) || p <= 0 || p > 2
                display('Clamping p');
                p = 1.0;
            end

            % Update
            delta(k) = delta(k) + p * dx(1);
            V(k)     = V(k) + p * dx(2);
        end

        if converged
            IterationCount(i,j) = iter;
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

levels = [5, 6, 7, 8, max_iter];
[C,h] = contour(delta_range, V_range, IterPlot, levels, 'k', 'LineWidth', 1.2);
clabel(C,h,'FontSize',12,'Color','k','LabelSpacing',600,...
       'BackgroundColor','none','EdgeColor','none');

colormap([0 0 0; jet(max_iter)])
caxis([-1 max_iter])
colorbar
xlabel('\theta_2 [rad]')
ylabel('V_2 [pu]')
if addTitle
    title('Region of Attraction of Iwamoto Method')
end
set(gca,'FontName','Times New Roman','FontSize',12)
saveas(gcf,'Iwamoto-RegionOfAttraction.png');
