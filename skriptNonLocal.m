% =========================================================================
% Nonlocal traffic flow model with uncertain speed 
% =========================================================================
% Goal: determine an inflow control u(t) such that the resulting outflow
% at the end of the road segment tracks a prescribed demand D(t) as
% closely as possible - even though the actual speed v(rho) is scattered
% by a random, additive perturbation epsilon (uncertain speed).
%
% Structure of this script (each MATLAB code section "%%" can be run
% individually):
%   Section 0 - Model parameters, kernel weights, forward simulation (MC)
%   Section 1 - Optimal control: deterministic reference model
%   Section 2 - Optimal control: smoothed "proxy" speed model
%   Section 3 - Optimal control: full stochastic model (MC objective) +
%               comparison of proxy vs. stochastic
%   Section 4 - Sensitivity analysis: error as a function of the spread
%               of the speed perturbation
%
% Core functions used by this script:
%   stochasticNonlocalModel            - pure forward simulation (1 realization)
%   stochasticNonlocalModel_control    - forward simulation, returns only
%                                         the time window relevant to the
%                                         objective (t in [3,T])
%   stochasticNonlocalModel_control_stochastic - as above, but averaged
%                                         over many speed realizations
%                                         (MC, internally via parfor)
%   stochasticNonlocalModel_forward    - forward simulation for several
%                                         realizations at once (for
%                                         evaluation/plots after optimization)
%   jacobiMeanClosed                   - expected demand D(t) from a
%                                         Jacobi-/OU-type demand process
%   computeNonlocalKernelWeights       - quadrature weights gamma for the
%                                         nonlocal kernel (precomputed
%                                         once, see Section 0)
% =========================================================================


%% Section 0: Model parameters, kernel weights, and forward simulation (MC)
% Spatio-temporal grid and the nonlocal kernel W (weighting of the speed
% over the "look-ahead" distance eta). gamma are the resulting precomputed
% 5-point Gauss quadrature weights, reused by every stochasticNonlocal-
% Model* function below instead of being recomputed on every single call
% (see computeNonlocalKernelWeights.m).

% No Figure connected, just to get an idea of the forward model

xrange = [0,1];
uFct = @(t) 0.2 + 0.05*sin(t);

dx = 0.05;
dt = dx/2;
T = 10;
u = uFct(0:dt:T);
v = @(rho) 1-rho.^2;
eta = 0.2;
W = @(x) 3/(2*eta)*(eta^2-x.^2);
gamma = computeNonlocalKernelWeights(dx, eta, W);

% The actual speed is scattered by an additive perturbation
% epsilon ~ U([-0.2,0.2]). MC realizations are forward-simulated under
% the fixed control u to look at the distribution of the resulting
% outflow (mean, median, 20%/80% quantile).
MC = 100;
epsilon = 0.4*rand(1,MC) - 0.2;
% +1: time grid covers (like u and D below) the endpoint t=T as well, see
% stochasticNonlocalModel.m.
outflow = zeros(round(T/dt)+1,MC);
tic()
for i=1:MC
    v = @(rho) max(0,1-rho.^2 + epsilon(i));
    [outflow(:,i),~] = stochasticNonlocalModel(xrange,u,dx,dt,T,v,gamma);
end
tid=toc()

% outflow(1,:) <-> t=0 is always 0 (initial condition, no outflow yet)
% and is skipped when plotting: outflow(2:end,:) <-> t=dt..T matches
% exactly the x-grid dt:dt:T.
figure
plot(dt:dt:T, mean(outflow(2:end,:),2), LineWidth=2)
hold on
plot(dt:dt:T, median(outflow(2:end,:),2), LineWidth=1.5, LineStyle="--")
plot(dt:dt:T, quantile(outflow(2:end,:),0.2,2), LineWidth=1.2, LineStyle=":")
plot(dt:dt:T, quantile(outflow(2:end,:),0.8,2), LineWidth=1.2, LineStyle="-.")
xlabel('time')
ylabel('outflow')
legend('Mean Outflow','Median Outflow', '20% quantile', '80% quantile','Location','best')
title('Outflow of nonlocal model')

% Optional (disabled): animate the spatial density profile rho(t,x) over time
% figure
% for i=1:10:(T/dt)
%     plot((xrange(1)+dx):dx:(xrange(2)-dx),rho(i,1:end-1),'LineWidth',2)
%     ylim([0,1])
%     pause(0.0001)
% end


%% Section 1: Optimal control for the deterministic reference problem
% D(t) is the expected demand, computed from a Jacobi-/OU-type process
% via jacobiMeanClosed. Optimization is only over t in [3,T] (the first
% 3 time units serve to settle transients, see DD). The speed here is the
% unperturbed reference function v(rho)=1-rho^2 (no proxy, no stochastic
% perturbation) - this run provides the deterministic baseline against
% which the following sections compete.

% No Figure connected, just to set the deterministic benchmark

d0 = 0.16;
kappa = 4;
theta_fun = @(t) 0.1 + 0.05*cos(t);
D = jacobiMeanClosed(0:dt:T, d0, kappa, theta_fun);
DD = D(3/dt+1:end)';

v = @(rho) 1-rho.^2;
objFct = @(u) dt*sum((DD-stochasticNonlocalModel_control(xrange,u,dx,dt,T,v,gamma)).^2);

tic()
[optInflow, residuum] = fmincon(objFct,0.2*ones(1,length(0:dt:T)));
tidOpt = toc()

% Comparison plot: optimal control, resulting outflow, demand
figure()
plot(0:dt:T,optInflow, 'LineWidth',1.2,'LineStyle',':')
hold on
% outflow(1) <-> t=0 is skipped (see above), outflow(2:end) <-> t=dt..T
outflowDet = stochasticNonlocalModel(xrange,optInflow,dx,dt,T,v,gamma);
plot(dt:dt:T,outflowDet(2:end),'LineWidth',2)
plot(0:dt:T, D, 'LineWidth',1.5,'LineStyle','--')
hold off
legend('Inflow','outflow','demand')
xlabel('time')
ylabel('inflow')
title('Comparison of optimal outflow and demand - proxy')


%% Section 2: Optimal control for a smoothed "proxy" speed model
% D and DD remain unchanged (same demand as in Section 1). Here v is
% replaced by a smoothed variant of 1-rho^2 with a small lower plateau
% (avoids v=0 near rho=1) - a deterministic "proxy" approximation of the
% actual, stochastically perturbed speed. Caution: this particular
% formula (threshold 0.2, factor 1.25) is only correctly calibrated for
% epsilon ~ U([-0.2,0.2]).

% No Figure connected, just to get an idea of the proxy model

v = @(rho) ...
    (1 - rho.^2).*(1 - rho.^2 >= 0.2) + ...
    ((1 - rho.^2) + 1.25*(0.2 - (1 - rho.^2)).^2).*(1 - rho.^2 < 0.2);

objFct = @(u) dt*sum((DD-stochasticNonlocalModel_control(xrange,u,dx,dt,T,v,gamma)).^2);

tic()
[optInflowProxy, residuumProxy] = fmincon(objFct,0.15*ones(1,length(0:dt:T)));
tidOptProxy = toc()

figure()
plot(0:dt:T,optInflowProxy, 'LineWidth',1.2,'LineStyle',':')
hold on
% outflow(1) <-> t=0 is skipped (see above), outflow(2:end) <-> t=dt..T
outflowProxyDet = stochasticNonlocalModel(xrange,optInflowProxy,dx,dt,T,v,gamma);
plot(dt:dt:T,outflowProxyDet(2:end),'LineWidth',2)
plot(0:dt:T, D, 'LineWidth',1.5,'LineStyle','--')
hold off
legend('Inflow','outflow','demand')
xlabel('time')
ylabel('inflow')
title('Comparison of optimal outflow and demand - proxy')


%% Section 3: Optimal control for the full stochastic model
% Instead of a single deterministic speed function, this uses an
% objective that averages the expected squared error over MC=200
% independent realizations of the perturbed speed v_vec{i}
% (stochasticNonlocalModel_control_stochastic, internally via parfor over
% the realizations). This is the "actual" stochastic objective, against
% which the deterministic proxy from Section 2 is compared below.

% Adjust the value of MC to change the number of Monte Carlo runs.

% FIGURE 7: left part


MC = 10;
v_vec = cell(MC,1);
epsilon = 0.4*rand(1,MC) - 0.2;
for i=1:MC
    v_vec{i} = @(rho) max(0,1-rho.^2 + epsilon(i));
end

objFctStoch = @(u)stochasticNonlocalModel_control_stochastic(xrange,u,dx,dt,T,v_vec,gamma,DD) ;

tic()
[optInflow_Stoch, residuum_Stoch] = fmincon(objFctStoch,0.15*ones(1,length(0:dt:T)));
tidOptStoch = toc()

% Both optimal controls (stochastic and proxy) are forward-simulated here
% on the same speed ensemble v_vec, for a fair comparison ("how does the
% control optimized in Section 2 with the proxy model perform when tested
% on the real stochastic model?").
forward_Out_Stoch = stochasticNonlocalModel_forward(xrange,optInflow_Stoch,dx,dt,T,v_vec,gamma);
forward_Out_Proxy = stochasticNonlocalModel_forward(xrange,optInflowProxy,dx,dt,T,v_vec,gamma);

figure()
plot(0:dt:T,optInflow_Stoch, 'LineWidth',1,'LineStyle','-','Color','r')
hold on
plot(0:dt:T,optInflowProxy, 'LineWidth',1,'LineStyle','-','Color','b')
% forward_Out_*(1,:) <-> t=0 is skipped (see above), (2:end,:) <-> t=dt..T
plot(dt:dt:T,mean(forward_Out_Stoch(2:end,:),2),'LineWidth',2,'Color','r','LineStyle',':')
plot(dt:dt:T,mean(forward_Out_Proxy(2:end,:),2),'LineWidth',2,'Color','b','LineStyle',':')
plot(0:dt:T, D, 'LineWidth',1.5,'LineStyle','--','Color','g')
hold off
legend('Inflow Stochastic','Inflow Proxy','Outflow Stochastic','Outflow Proxy','Demand','Location','best')
xlabel('time')
ylabel('inflow')
title('Comparison of optimal outflow and demand - Stochastic vs. Proxy')

% Error in the objective: stochastically optimized vs. proxy-optimized
% control, both evaluated under the true stochastic ensemble.
% forward_Out_*(k,:) <-> t=(k-1)*dt, hence 3/dt+1:end instead of 3/dt:end,
% so that this is exactly aligned with D(3/dt+1:end) (t=3..T).
disp('Error Stochastic')
err_Stoch = dt*sum(mean((D(3/dt+1:end)' - forward_Out_Stoch(3/dt+1:end,:)).^2,2))
disp('Error Proxy')
err_Proxy = dt*sum(mean((D(3/dt+1:end)'- forward_Out_Proxy(3/dt+1:end,:)).^2,2))




%% Section 4: Sensitivity analysis - error vs. spread of the speed perturbation
% Repeats Section 2+3 (proxy and stochastic optimization) for decreasing
% spreads tau=2^qqq of the speed perturbation, to examine how quickly the
% "excess cost" (the proxy's extra cost relative to the stochastic
% solution) vanishes as the uncertainty in the speed goes to 0.
% v_helpProxy is the generalized, tau-parametrized version of the proxy
% speed from Section 2.

% Adjust the value of MC to change the number of Monte Carlo runs. Be
% careful, increasing MC has a significant impact on the runtime!

% FIGURE 7: Right part
k = -5:-1;   % exponents: tau = 2^k, from k=-5 (narrow) to k=-1 (wide)

meanErrorStoch = zeros(length(k),1);
meanErrorProxy = zeros(length(k),1);

v_helpProxy = @(rho,tau) ...
    ( (1 - rho.^2 >= tau) .* (1 - rho.^2) ) + ...
    ( (1 - rho.^2 >= 0 & 1 - rho.^2 < tau) .* ...
      ((1 - rho.^2) + (tau - (1 - rho.^2)).^2 ./ (4*tau)) );
idx = 0;
for qqq = k
    qqq
    idx = idx+1;

    % --- Proxy optimization for the current spread tau=2^qqq ---
    v = @(rho) v_helpProxy(rho,2^(qqq));
    objFct = @(u) dt*sum((DD-stochasticNonlocalModel_control(xrange,u,dx,dt,T,v,gamma)).^2);

    tic()
    [optInflowProxy, residuumProxy] = fmincon(objFct,0.15*ones(1,length(0:dt:T)));
    tidOptProxy = toc()

    % --- Stochastic optimization for the same tau: epsilon spread band
    % scales proportionally to tau, MC=200 fresh realizations ---
    MC = 10;
    v_vec = cell(MC,1);
    epsilon = 2*(2^(qqq))*rand(1,MC)-2^(qqq);
    for i=1:MC
        v_vec{i} = @(rho) max(0,1-rho.^2 + epsilon(i));
    end
    objFctStoch = @(u)stochasticNonlocalModel_control_stochastic(xrange,u,dx,dt,T,v_vec,gamma,DD) ;

    tic()
    [optInflow_Stoch, residuum_Stoch] = fmincon(objFctStoch,0.15*ones(1,length(0:dt:T)));
    tidOptStoch = toc()

    % --- Error of both solutions, evaluated under the true stochastic
    % ensemble of this iteration (see comment on err_Stoch/err_Proxy in
    % Section 3: 3/dt+1:end instead of 3/dt:end) ---
    forward_Out_Stoch = stochasticNonlocalModel_forward(xrange,optInflow_Stoch,dx,dt,T,v_vec,gamma);
    forward_Out_Proxy = stochasticNonlocalModel_forward(xrange,optInflowProxy,dx,dt,T,v_vec,gamma);

    meanErrorStoch(idx) = dt*sum(mean((D(3/dt+1:end)' - forward_Out_Stoch(3/dt+1:end,:)).^2,2));
    meanErrorProxy(idx) = dt*sum(mean((D(3/dt+1:end)' - forward_Out_Proxy(3/dt+1:end,:)).^2,2));
end

% --- Result 1: error curves of the stochastic and proxy solution over tau ---
figure
plot(k, log2(meanErrorStoch), 'LineWidth',2)
hold on
plot(k, log2(meanErrorProxy), 'LineWidth',2)
hold off
xlabel('k')
ylabel('log2(error)')
legend('Stochastic','Proxy')
title('Error evolution for smaller supports of \varepsilon')

% --- Result 2: "excess cost" = difference of the two error curves ---
figure
plot(k, log2(abs(meanErrorStoch-meanErrorProxy)), 'LineWidth',2)
xlabel('k')
ylabel('log2(error)')
legend('Excess cost')
title('Error evolution for smaller supports of \varepsilon in the nonlocal model')

% --- Result 3: excess cost with a linear fit (convergence rate in tau) ---
figure

y = log2(abs(meanErrorStoch - meanErrorProxy));

y = y(1:end-1);
k1 = k(1:end-1);

% Plot the data
plot(k1, y, 'o-', 'LineWidth', 2, 'MarkerSize', 7)
hold on

% Linear best fit
p = polyfit(k1, y, 1);
yFit = polyval(p, k1);

plot(k1, yFit, '--', 'LineWidth', 2)

% Slope
slope = p(1);

xlabel('$k$', 'Interpreter', 'latex')
ylabel('$\log_2(\mathrm{error})$', 'Interpreter', 'latex')

legend( ...
    'Excess cost', ...
    sprintf('Linear fit, slope = %.3f', slope), ...
    'Location', 'best' ...
)

title('Error evolution for smaller supports of $\varepsilon$ in the nonlocal model', ...
    'Interpreter', 'latex')

grid on
box on
