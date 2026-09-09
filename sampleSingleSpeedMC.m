function [L1_error_mean,L2_error_mean,L3_error_mean,mean_out,mean_X,lambda,X,u,quantile20,quantile80] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU,plotBool,ii,velocityShift,randeinzug,outputMode,applyBoundary,transportMethod,velocityShiftControl,lambdaIn,Xin,tMin,tMax)
% Samples MCruns realizations of the Jacobi demand process and the
% random speed lambda, computes the corresponding optimal inflow control
% u, forward-simulates the resulting outflow, and returns the L1/L2/L3
% error against the demand (trapezoidal rule over [randeinzug,T-randeinzug]).
% For the paper we only use the L2-error.
%
%   outputMode           'piecewise' (default) or 'continuous', see
%                         computeOptimalU.m. dtU is the control-grid cell
%                         width for 'piecewise'; pass dtU=dt for 'continuous'.
%   applyBoundary         apply the Lambda(t) boundary restriction (3.2)
%                         to the control (default true), see computeOptimalU.m.
%   transportMethod       'trajec' (default; characteristics, exact) or
%                         'upwind' (grid-based upwind scheme), see
%                         transportUpwind_trajec.m / transportUpwind.m.
%   velocityShiftControl  [lambdaMin,lambdaMax] assumed when *computing*
%                         the control (default: same as velocityShift,
%                         the range the outflow is actually *simulated*
%                         under). Passing a different (e.g. near-
%                         degenerate) interval here reproduces the old
%                         "proxy" control evaluated against the true
%                         stochastic dynamics.
%   lambdaIn, Xin         reuse existing lambda/demand realizations
%                         instead of sampling new ones (default: sample
%                         fresh). Pass both to evaluate a different
%                         control on the same random paths as an earlier
%                         call, for a paired (fair) comparison.
%   tMin, tMax            absolute error-evaluation window [tMin,tMax]
%                         (default: [randeinzug, T-randeinzug], i.e. the
%                         symmetric trim randeinzug already describes).
%                         Pass explicitly for an asymmetric window - e.g.
%                         the paper's observation window I_obs=[1/lambdaMIN,T]
%                         (Sec. 3.1): unlike a symmetric trim, this
%                         excludes only the initial segment [0,1/lambdaMIN)
%                         where the outflow is provably 0 regardless of
%                         the control (no realization can have an arrival
%                         yet), instead of also cutting away genuine
%                         boundary behavior at t=T.

if nargin < 15 || isempty(outputMode);          outputMode = 'piecewise';   end
if nargin < 16 || isempty(applyBoundary);       applyBoundary = true;       end
if nargin < 17 || isempty(transportMethod);     transportMethod = 'trajec'; end
if nargin < 18 || isempty(velocityShiftControl); velocityShiftControl = velocityShift; end
if nargin < 19; lambdaIn = []; end
if nargin < 20; Xin = []; end
if nargin < 21 || isempty(tMin); tMin = randeinzug;   end
if nargin < 22 || isempty(tMax); tMax = T - randeinzug; end

t_grid = 0:dt:T;

if isempty(lambdaIn) || isempty(Xin)
    % Sample Jacobi demand and speed
    [t, X] = simulateJacobiTimeDep(d0, kappa, theta_fun, sigma, dt, T, MCruns);
    lambda = velocityShift(1)+(velocityShift(2)-velocityShift(1))*betarnd(alpha,beta,MCruns,1);
else
    % Reuse existing realizations (paired comparison against a previous call)
    t = t_grid;
    X = Xin;
    lambda = lambdaIn;
end

t_gridU = 0:dtU:T;

u = computeOptimalU(t_gridU, d0, kappa, theta_fun, 5*10^-4, velocityShiftControl, alpha, beta, applyBoundary, outputMode);

% u_t: expand the (piecewise-constant or fine-grid) control u onto t_grid.
u_t = zeros(size(t_grid));
N = length(u);
for i = 1:N
    if i < length(t_gridU)
        idx = (t_grid >= t_gridU(i)) & (t_grid < t_gridU(i+1));
    else
        idx = t_grid >= t_gridU(i);
    end
    u_t(idx) = u(i);
end
u_t(t_grid >= t_gridU(end)) = u(end);   % exact endpoint t=T

% Outflow
switch transportMethod
    case 'trajec'
        out = transportUpwind_trajec(lambda, u_t, dt, T);
    case 'upwind'
        out = transportUpwind(lambda, u_t, dt, T);
end

% Plotting (single realization + mean over all realizations)
if plotBool
    colors = lines(MCruns);
    figure(2*ii-1);
    hold on;
    for k = 1:min(MCruns,1)
        plot(t, out(k,:), 'Color', colors(k,:), 'LineWidth', 1.5);
        plot(t, X(k,:), '--', 'Color', colors(k,:), 'LineWidth', 1);
    end
    xlabel('t');
    ylabel('Outflow / Demand');
    title('Outflow vs. Jacobi-Demand (Realizations)');
    legend_entries = cell(1,min(MCruns,1)*2);
    for k = 1:2:(2*min(MCruns,1))
        legend_entries{k} = ['MC Outflow ' num2str(k)];
    end
    for k = 2:2:(2*min(MCruns,1))
        legend_entries{k} = ['MC Demand ' num2str(k)];
    end
    legend(legend_entries);
    hold off;

    figure(2*ii);
    hold on;
    plot(t, mean(out,1), '-b', 'LineWidth', 2);
    plot(t, mean(X,1), '--r', 'LineWidth', 2);
    xlabel('t');
    ylabel('Outflow / Demand');
    title('Outflow vs. Jacobi-Demand (Realizations)');
    legend('Mean Outflow','Mean Demand');
    grid on;
    hold off;
end

% L1/L2/L3 error per path, trapezoidal rule over the evaluation window
% [tMin, tMax]
tminIdx = tMin/dt+1;
tmaxIdx = tMax/dt+1;
Xerr = X(:,tminIdx:tmaxIdx);
outErr = out(:,tminIdx:tmaxIdx);
err1 = zeros(MCruns,1);
err2 = zeros(MCruns,1);
err3 = zeros(MCruns,1);
for k = 1:MCruns
    err1(k) = trapz(t(tminIdx:tmaxIdx), abs(Xerr(k,:) - outErr(k,:)));
    err2(k) = trapz(t(tminIdx:tmaxIdx), (Xerr(k,:) - outErr(k,:)).^2);
    err3(k) = trapz(t(tminIdx:tmaxIdx), (Xerr(k,:) - outErr(k,:)).^3);
end
L1_error_mean = mean(err1);
L2_error_mean = mean(err2);
L3_error_mean = mean(err3);

mean_out = mean(out, 1);
mean_X   = mean(X, 1);
quantile20 = quantile(out,0.2,1);
quantile80 = quantile(out,0.8,1);
