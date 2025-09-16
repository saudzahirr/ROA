clear all; clc; format long;

global addTitle;

%% -------------------------------
%  2-BUS SYSTEM
% -------------------------------
Z12 = 0 + 1j*0.1;         % line impedance
Y12 = -1/Z12;
Y_bus = [ -Y12,   Y12;
           Y12,  -Y12 ];

G = real(Y_bus);
B = imag(Y_bus);

% Bus types: 1 = slack, 0 = PQ
bus_type = [1; 0];
pq_bus_id = find(bus_type == 0);

%% -------------------------------
%  PQ Grid (parameter space sweep)
% -------------------------------
P_range = linspace(0.01, 6.0, 256);   % pu load (0–600 MW)
Q_range = linspace(0.01, 3.0, 256);   % pu load (0–300 MVar)

nP = length(P_range);
nQ = length(Q_range);

Result = NaN(nP,nQ);

tolerance = 1e-5;
max_iter = 25;

%% -------------------------------
%  Newton–Raphson Sweep in PQ space
% -------------------------------
for i = 1:nP
    for j = 1:nQ
        % Scheduled load (negative injection convention)
        P_spec = [0; -P_range(i)];
        Q_spec = [0; -Q_range(j)];

        % Initial guess
        V = [1.0; 1.0];      
        delta = [0.0; 0.0];

        iter = 0;
        converged = false;

        while iter < max_iter
            iter = iter + 1;

            % Complex voltages
            Vc = V .* exp(1j*delta);

            % Power injections
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

            % Jacobian for bus 2
            k = pq_bus_id;
            Vk = V(k);
            Pk = P_calc(k); 
            Qk = Q_calc(k);

            dP_dtheta = -Qk - (B(k,k)*Vk^2);
            dP_dV     = (Pk/Vk) + G(k,k)*Vk;
            dQ_dtheta = Pk - G(k,k)*Vk^2;
            dQ_dV     = (Qk/Vk) - B(k,k)*Vk;

            J = [dP_dtheta, dP_dV;
                 dQ_dtheta, dQ_dV];

            dx = J \ mismatch;

            % Update
            delta(k) = delta(k) + dx(1);
            V(k)     = V(k) + dx(2);
        end

        if converged
            Result(i,j) = 1; % Solvable
        else
            Result(i,j) = 0; % Unsolvable
        end
    end
end

%% -------------------------------
%  Analytical Solvability Boundary
% -------------------------------
X12 = 0.1; 
Bpos = 1/X12;       % positive susceptance
P_boundary = linspace(0,6.0,500);   % in pu
Q_boundary = -(P_boundary.^2)/Bpos + (Bpos/4);  % in pu

%% -------------------------------
%  Plot PQ solvability region + boundary
% -------------------------------
S_base = 100;  % MVA base

figure('Color',[1 1 1])
imagesc(P_range*S_base, Q_range*S_base, Result')  % axes: P on x, Q on y
set(gca,'YDir','normal')

hold on
plot(P_boundary*S_base, Q_boundary*S_base, 'k-', 'LineWidth', 2) % analytical boundary
text(230,210,'\bf\Sigma','FontSize',18,'FontName','Times New Roman') % label Σ

colormap([1 0 0; 0 1 0])  % red=unsolvable, green=solvable
xlabel('Real power load P (MW)','FontSize',12,'FontName','Times New Roman')
ylabel('Reactive power load Q (MVar)','FontSize',12,'FontName','Times New Roman')
if addTitle
    title('PQ Solvability Region (2-Bus)','FontSize',12,'FontName','Times New Roman')
end

set(gca,'FontName','Times New Roman','FontSize',12)
saveas(gcf,'SolvabilityBoundary.png');
