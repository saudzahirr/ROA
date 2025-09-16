clear all; clc; format long;

global addTitle;

%% -------------------------------
%  2-BUS SYSTEM - FIXED POINT ITERATION
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

% Scheduled powers
P_spec = [0; -0.9];   % P2 = -0.9 pu (load)
Q_spec = [0; -0.6];   % Q2 = -0.6 pu (load)

% Bus types: 1 = slack, 0 = PQ
bus_type = [1; 0];

% Indices
pq_bus_id = find(bus_type == 0);
slack_id  = find(bus_type == 1);

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
max_iter = 100;
relaxation_factor = 0.7;  % Under-relaxation for stability

%% -------------------------------
%  Fixed Point Iteration Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        try
            % Initial guess
            V = [1.0; V_range(i)];
            delta = [0.0; delta_range(j)];

            k = pq_bus_id; % PQ bus index (bus 2)

            iter = 0;
            converged = false;

            while iter < max_iter
                iter = iter + 1;

                % Store previous values
                V_old = V(k);
                delta_old = delta(k);

                % Complex voltages
                Vc = V .* exp(1j*delta);

                % Fixed point iteration for PQ bus
                % From power balance: P + jQ = V * conj(I) = V * conj(Y * V)
                % Rearrange to get fixed point form: V_k = f(V_k, V_others)

                S_specified = P_spec(k) + 1j*Q_spec(k);

                % Calculate current injection from other buses
                I_from_others = 0;
                for m = 1:n_bus
                    if m ~= k
                        I_from_others = I_from_others + Y_bus(k,m) * Vc(m);
                    end
                end

                % Fixed point equation: V_k = (S_k* / I_k*) where I_k is total injection
                % S_k = V_k * I_k* => I_k* = S_k* / V_k*
                % I_k = Y_kk * V_k + I_from_others
                % So: Y_kk * V_k = I_k - I_from_others = S_k* / V_k* - I_from_others

                if abs(Vc(k)) > 1e-10
                    % Method 1: Direct substitution
                    I_total_conj = conj(S_specified) / conj(Vc(k));
                    V_new_complex = (I_total_conj - I_from_others) / Y_bus(k,k);

                    % Apply under-relaxation
                    V_new_complex = (1 - relaxation_factor) * Vc(k) + relaxation_factor * V_new_complex;

                    V_new = abs(V_new_complex);
                    delta_new = angle(V_new_complex);
                else
                    break; % Avoid division by zero
                end

                % Alternative Method 2: Separate real and imaginary parts
                % Can be more stable for some cases
                if iter > 50  % Switch methods if not converging
                    % Calculate power injections
                    I_total = Y_bus * Vc;
                    S_calc = Vc .* conj(I_total);
                    P_calc = real(S_calc);
                    Q_calc = imag(S_calc);

                    % Power errors
                    dP = P_spec(k) - P_calc(k);
                    dQ = Q_spec(k) - Q_calc(k);

                    % Simple gradient-based update
                    alpha = 0.1;  % Step size
                    delta_new = delta_old + alpha * dP / (V_old^2 + 1e-6);
                    V_new = V_old + alpha * dQ / (V_old + 1e-6);
                end

                % Apply relaxation and bounds
                V(k) = (1 - relaxation_factor) * V_old + relaxation_factor * V_new;
                delta(k) = (1 - relaxation_factor) * delta_old + relaxation_factor * delta_new;

                % Bound variables to prevent divergence
                V(k) = max(0.1, min(V(k), 3.0));
                delta(k) = max(-pi, min(delta(k), pi));

                % Check convergence
                dV = abs(V(k) - V_old);
                ddelta = abs(delta(k) - delta_old);

                if max(dV, ddelta) < tolerance
                    % Verify solution by checking power balance
                    Vc_final = V .* exp(1j*delta);
                    I_final = Y_bus * Vc_final;
                    S_final = Vc_final .* conj(I_final);
                    P_error = abs(P_spec(k) - real(S_final(k)));
                    Q_error = abs(Q_spec(k) - imag(S_final(k)));

                    if max(P_error, Q_error) < tolerance * 10
                        converged = true;
                        break;
                    end
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
    levels = [14 17:25];
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
    title('Region of Attraction of Fixed Point Iteration')
end

% Set font to Times New Roman, size 12
set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf, 'FixedPoint-RegionOfAttraction.png');
