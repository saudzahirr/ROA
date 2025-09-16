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

% Scheduled powers
S_spec = [0; -(0.9 + 1j*0.6)];   % P+jQ (PQ load at bus 2)

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
max_iter = 25;   % Jacobi usually even slower

%% -------------------------------
%  Jacobi Sweep
% -------------------------------
for i = 1:nV
    for j = 1:nd
        % Initial guess
        V = zeros(n_bus,1);
        V(1) = 1.0;  % Slack fixed
        V(2) = V_range(i) * exp(1j*delta_range(j)); % initial guess for bus 2

        iter = 0;
        converged = false;

        while iter < max_iter
            iter = iter + 1;

            V_new = V;  % store new values separately

            % Update PQ bus (bus 2 here)
            k = pq_bus_id;
            Vk_old = V(k);

            % Jacobi update formula (same as GS but uses only old voltages):
            sum_term = 0;
            for m = 1:n_bus
                if m ~= k
                    sum_term = sum_term + Y_bus(k,m)*V(m);
                end
            end

            V_new(k) = (1/Y_bus(k,k)) * ( (conj(S_spec(k))/conj(Vk_old)) - sum_term );

            % Convergence check
            if abs(V_new(k) - V(k)) < tolerance
                converged = true;
                break;
            end

            % Update for next iteration (simultaneous update)
            V = V_new;
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
    levels = 5:10;
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
    title('Region of Attraction of Jacobi Method')
end

% Set font to Times New Roman, size 12
set(gca,'FontName','Times New Roman','FontSize',12)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',12)

saveas(gcf, 'Jacobi-RegionOfAttraction_Jacobi.png');
