% =========================================================================
% Optimal Inflow Control for Transport Equations with Uncertain
% Velocities and Demand - numerical experiments
% =========================================================================
% Reproduces Figures 2-6 from Goettlich & Schillinger, "Optimal Inflow
% Control for Transport Equations with Uncertain Velocities and Demand"
% (arXiv:2609.01291), Sections 6.2-6.4. (Figure 7, the nonlocal-model
% comparison, is produced separately by skriptNonLocal.m.)
%
%   Section "Figure 2 & 3" - temporal discretization: convergence of
%                            piecewise-constant controls (paper Sec. 6.2)
%   Section "Figure 4 & 5" - velocity uncertainty and the mean-velocity
%                            proxy (paper Sec. 6.3)
%   Section "Figure 6"     - boundary effects of the fixed observation
%                            window (paper Sec. 6.4)
%
% Sections must be run in order (each reuses parameters set by the
% previous one, in particular theta_fun/d0/kappa/sigma/dt).
% =========================================================================


%% Figure 2 & 3: Temporal discretization - convergence of piecewise-constant controls
% Paper Section 6.2. lambda ~ U([1,3]), T=16, interior window J=[2,T-2],
% control-grid cell length |Pi_n| = 2^k for k=-4,...,1.

T = 16;
MCruns = 25000;

alpha = 1;
beta = 1;

lambdaMIN = 1;
lambdaMAX = 3;
velocityShift = [lambdaMIN, lambdaMAX];

theta_fun = @(t) 4*(0.5+0.25*sin(pi*t));
d0 = 4*0.4;
kappa = 4;
sigma = 0.15;
dt = 0.0005;

randeinzug = 2;   % interior evaluation window J = [randeinzug, T-randeinzug] = [2, T-2]
plotBool = 0;

dtU = 2.^(1:-1:-4);   % |Pi_n| = 2^k, k = -4,...,1

l2   = zeros(length(dtU),1);   % piecewise-constant
l2P  = zeros(length(dtU),1);   % piecewise-constant + proxy
l2C  = zeros(length(dtU),1);   % continuous
l2PC = zeros(length(dtU),1);   % continuous + proxy

u_all = cell(length(dtU),1);
meanOutPWC = cell(length(dtU),1);

tic()
for i = 1:length(dtU)
    i

    % Piecewise-constant optimal control
    [~,L2_error_mean,~,meanOutPWC{i},~,lambda,X,u_all{i},~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU(i),plotBool,i,velocityShift,randeinzug);
    l2(i) = L2_error_mean;

    % Piecewise-constant + proxy (reuses the lambda, X realizations from above)
    [~,L2_error_cont,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU(i),plotBool,i,velocityShift,randeinzug,'piecewise',true,'upwind',[2-1000*eps,2+1000*eps],lambda,X);
    l2P(i) = L2_error_cont;

    % Continuous-time optimal control and continuous + proxy do not depend
    % on dtU, so they are computed once (i==1) and broadcast across the sweep.
    if i == 1
        [~,L2_error_cont,~,mean_out,~,lambda,X,u,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind');
        l2C(:) = L2_error_cont;

        [~,L2_error_contP,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind',[2-1000*eps,2+1000*eps]);
        l2PC(:) = L2_error_contP;
    end
end
tid = toc()

% Figure 3: log-log excess cost H(u^(n),*)-H(u^*) and H(ubar^(n))-H(ubar)
% for decreasing control-grid cell length.
figure
y2 = log2(abs(l2C  - l2));      % continuous vs. piecewise-constant
y3 = log2(abs(l2PC - l2P));     % continuous+proxy vs. piecewise-constant+proxy
hold on
h2 = plot(log2(dtU), y2, LineWidth=1.5);
h3 = plot(log2(dtU), y3, LineWidth=1.5, LineStyle="--");
x = log2(dtU);
idxFit = x >= -4 & x <= 0;
p2 = polyfit(x(idxFit), y2(idxFit), 1);
p3 = polyfit(x(idxFit), y3(idxFit), 1);
xfit = linspace(-4, 0, 100);
plot(xfit, polyval(p2,xfit), ':', 'Color', h2.Color, 'LineWidth', 0.8)
plot(xfit, polyval(p3,xfit), ':', 'Color', h3.Color, 'LineWidth', 0.8)
xlabel('$\log_2(|\Pi_n|)$', 'Interpreter', 'latex')
ylabel('$\log_2(\mathrm{error})$', 'Interpreter', 'latex')
title('Logarithmic error for decreasing the control grid cell length', ...
    'Interpreter', 'latex', 'FontName', 'Arial')
legend( ...
    '$H(u^{(n),*})-H(u^*)$', ...
    '$H(\bar{u}^{(n)})-H(\bar{u})$', ...
    ['$\mathrm{fit}: s=' sprintf('%.3f',p2(1)) '$'], ...
    ['$\mathrm{fit}: s=' sprintf('%.3f',p3(1)) '$'], ...
    'Interpreter', 'latex', 'Location', 'best')
hold off

% Figure 2 (left): optimal inflow for decreasing control-interval length
leg_entries = cell(1,length(1:2:length(u_all))+1);
figure
hold on
for k = 1:2:length(u_all)
    ui = u_all{k};
    t = linspace(0, T, length(ui));
    stairs(t, ui, 'LineWidth', 1.2);
    leg_entries{(k+1)/2} = ['k = ' num2str(log2(dtU(k)))];
end
plot(0:dt:T,u,'LineWidth',2)
leg_entries{end} = 'continuous';
hold off
xlabel('t'); ylabel('inflow');
title('Optimal inflow for varying lengths of the control intervals');
legend(leg_entries, 'Location', 'best');
grid on;

% Figure 2 (right): mean demand and optimal outflows for decreasing control-interval length
figure
hold on
for k = 1:2:length(meanOutPWC)
    outi = meanOutPWC{k};
    t = linspace(0, T, length(outi));
    stairs(t, outi, 'LineWidth', 1.2);
end
plot(0:dt:T,mean_out,'LineWidth',2,'LineStyle',':')
plot(dt:dt:T, jacobiMeanClosed(dt:dt:T, d0, kappa, theta_fun),'LineWidth',2,'LineStyle','--')
hold off
xlabel('t'); ylabel('outflow');
title('Mean demand and optimal outflows for varying lengths of the control intervals');
leg_entries{end+1} = 'mean demand';
legend(leg_entries, 'Location', 'best');
grid on;


%% Figure 4 & 5: Velocity uncertainty and the mean-velocity proxy
% Paper Section 6.3. Mean velocity fixed at 2, control-grid cell length
% |Pi_n|=1/2 fixed, and the support of lambda ~ U([lambdaMIN,lambdaMAX])
% shrinks symmetrically around 2 with Var(lambda) proportional to 2^k,
% k=-6,...,3. J=[2,T-2] as in Section 6.2 (reused: T=16 there too, so the
% interior interval remains valid for every velocity distribution below).

T = 16;
MCruns = 25000;

alpha = 1;
beta = 1;

theta_fun = @(t) 4*(0.5+0.25*sin(pi*t));
d0 = 4*0.4;
kappa = 4;
sigma = 0.15;
dt = 0.0005;

randeinzug = 2;
plotBool = 0;

dtU = 0.5;          % |Pi_n| = 1/2, fixed
shifts = 3:-1:-6;   % Var(lambda) proportional to 2^shifts

lambdaMIN = 2 - 0.5*sqrt(2.^shifts);
lambdaMAX = 2 + 0.5*sqrt(2.^shifts);

l2   = zeros(length(shifts),1);   % piecewise-constant
l2P  = zeros(length(shifts),1);   % piecewise-constant + proxy
l2C  = zeros(length(shifts),1);   % continuous
l2PC = zeros(length(shifts),1);   % continuous + proxy
l2D  = zeros(length(shifts),1);   % continuous, deterministic velocity (reference)
l2PD = zeros(length(shifts),1);   % piecewise-constant, deterministic velocity (reference)

meanOutC = cell(length(shifts),1);

tic()
for i = 1:length(shifts)
    i

    velocityShift = [lambdaMIN(i), lambdaMAX(i)];

    % Piecewise-constant optimal control
    [~,L2_error_mean,~,~,~,lambda,X,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU,plotBool,i,velocityShift,randeinzug);
    l2(i) = L2_error_mean;

    % Piecewise-constant + proxy
    [~,L2_error_cont,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU,plotBool,i,velocityShift,randeinzug,'piecewise',true,'upwind',[2-1000*eps,2+1000*eps],lambda,X);
    l2P(i) = L2_error_cont;

    % Continuous-time optimal control
    [~,L2_error_cont,~,meanOutC{i},~,lambda,X,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind');
    l2C(i) = L2_error_cont;

    % Continuous + proxy
    [~,L2_error_contP,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind',[2-1000*eps,2+1000*eps]);
    l2PC(i) = L2_error_contP;

    % Deterministic-velocity reference (lambda essentially fixed at 2):
    % independent of the shift, so only computed once (i==1) and
    % broadcast across the sweep.
    if i == 1
        [~,L2_error_contD,~,mean_outD,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,[2-1000*eps,2+1000*eps],randeinzug,'continuous',true,'upwind');
        l2D(:) = L2_error_contD;

        [~,L2_error_contPD,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU,plotBool,i,[2-1000*eps,2+1000*eps],randeinzug);
        l2PD(:) = L2_error_contPD;
    end
end
tid = toc()

% Figure 4 (right): logarithmic error evolution for Var(lambda) -> 0
figure
hold on
plot(shifts, log2(l2D),  'LineWidth',1.5, 'LineStyle',':')
plot(shifts, log2(l2C),  'LineWidth',1.5)
plot(shifts, log2(l2PC), 'LineWidth',1.5)
plot(shifts, log2(l2PD), 'LineWidth',1.5, 'LineStyle','--')
plot(shifts, log2(l2),   'LineWidth',1.5, 'LineStyle','--')
plot(shifts, log2(l2P),  'LineWidth',1.5, 'LineStyle','--')
legend('Continuous deterministic \lambda','Continuous','Continuous + Proxy','Piecewise-Constant deterministic \lambda','Piecewise-Constant','Piecewise-Constant + Proxy',Location='best')
xlabel('log2(Var(\lambda))')
ylabel('log2(error)')
title('Logarithmic error evolution for Var(\lambda) -> 0')
hold off

% Figure 4 (left): mean demand and optimal outflows for decreasing variance
figure
hold on
for k = 1:2:length(meanOutC)
    outi = meanOutC{k};
    t = linspace(0, T, length(outi));
    stairs(t, outi, 'LineWidth', 1.2);
end
plot(0:dt:T,mean_outD,'LineWidth',2,'LineStyle',':')
plot(dt:dt:T, jacobiMeanClosed(dt:dt:T, d0, kappa, theta_fun),'LineWidth',2,'LineStyle','--')
hold off
xlabel('t'); ylabel('outflow');
title('Mean demand and optimal outflows for decreasing variance in the transport velocity');
grid on;

% Figure 5: excess cost H(ubar)-H(u^*) and H(ubar^(n))-H(u^(n),*) vs. Var(lambda)
figure
y1 = log2(abs(l2PC - l2C));   % continuous+proxy vs. continuous
y2 = log2(abs(l2P  - l2));    % piecewise-constant+proxy vs. piecewise-constant
idxFit = shifts >= -3 & shifts <= 3;
h1 = plot(shifts, y1, LineWidth=2);
hold on
h2 = plot(shifts, y2, LineWidth=2, LineStyle="--");
p1 = polyfit(shifts(idxFit), y1(idxFit), 1);
p2 = polyfit(shifts(idxFit), y2(idxFit), 1);
xfit = linspace(-3, 3, 100);
plot(xfit, polyval(p1, xfit), ':', 'Color', h1.Color, 'LineWidth', 1.3)
plot(xfit, polyval(p2, xfit), ':', 'Color', h2.Color, 'LineWidth', 1.3)
xlabel('$\log_2(\mathrm{Var}(\lambda))$', 'Interpreter', 'latex')
ylabel('$\log_2(\mathrm{error})$', 'Interpreter', 'latex')
title('Logarithmic error for decreasing variance in $\lambda$', ...
    'Interpreter', 'latex', 'FontName', 'Arial')
legend( ...
    '$\log(H(\bar{u})-H(u^*))$', ...
    '$\log(H(\bar{u}^{(n)})-H(u^{(n),*}))~~$', ...
    ['$\mathrm{fit}: slope=' sprintf('%.3f',p1(1)) '$'], ...
    ['$\mathrm{fit}: slope=' sprintf('%.3f',p2(1)) '$'], ...
    'Interpreter', 'latex', 'Location', 'best')
xlim([-3, 3])   % display window matches the paper's Figure 5 (data/fit still use shifts=-6..3)
hold off


%% Figure 6: Boundary effects of the fixed observation window
% Paper Section 6.4. lambda ~ U([1,3]), T=6 (chosen small to emphasize the
% boundary region), comparing the corrected optimal control u*(t) against
% the "unconditioned interior expression" E_lambda[m(t+1/lambda)]. The
% latter is obtained via the *_wrongBoundary functions, i.e. without
% restricting to the admissible arrival-time set Lambda(t) from (3.2).
% Reuses theta_fun, d0, kappa, sigma, dt, alpha, beta from the Figure 2 & 3
% setup above.

T = 6;
MCruns = 25000;

lambdaMIN = 1;
lambdaMAX = 3;
velocityShift = [lambdaMIN, lambdaMAX];

randeinzug = 0;
ii = 1;

tic()
[~,~,~,~,~,~,~,u,~,~]   = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,ii,velocityShift,randeinzug,'continuous',true,'upwind');
[~,~,~,~,~,~,~,uWB,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,ii,velocityShift,randeinzug,'piecewise',false,'upwind');
tid = toc()

figure
plot(0:dt:(T-(1/3)),u(1:round(5.66666/dt)+1),'LineWidth',2)
hold on
plot(0:dt:(T-1/3),uWB(1:round(5.66666/dt)+1),'LineWidth',2,'LineStyle','--')
legend('optimal','interior')
xlim([0 5.6666666])
xlabel('time')
ylabel('controls')
title('Comparison of optimal control and unconditioned interior control')

% --- Figure 6 (right): convergence including boundary effects ---
% Same dtU-sweep and error/fit structure as Figure 3, but evaluated over
% the full objective (randeinzug=0, i.e. no interior trimming) at T=6
% instead of the interior-only window from Figure 2/3's T=16 setting.
% Paper: "the numerical experiment exhibits a convergence rate close to
% 2 for the full objective as well" (Sec. 6.4, Lemma 4.6).

randeinzug = 0;       % full objective H(u) over [0,T], no interior trimming
dtU = 2.^(1:-1:-4);   % |Pi_n| = 2^k, k = -4,...,1, same range as Figure 3

l2   = zeros(length(dtU),1);   % piecewise-constant
l2P  = zeros(length(dtU),1);   % piecewise-constant + proxy
l2C  = zeros(length(dtU),1);   % continuous
l2PC = zeros(length(dtU),1);   % continuous + proxy

tic()
for i = 1:length(dtU)
    i

    [~,L2_error_mean,~,~,~,lambda,X,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU(i),plotBool,i,velocityShift,randeinzug);
    l2(i) = L2_error_mean;

    [~,L2_error_cont,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dtU(i),plotBool,i,velocityShift,randeinzug,'piecewise',true,'upwind',[2-1000*eps,2+1000*eps],lambda,X);
    l2P(i) = L2_error_cont;

    % Continuous-time optimal control and continuous + proxy do not depend
    % on dtU, so they are computed once (i==1) and broadcast across the sweep.
    if i == 1
        [~,L2_error_cont,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind');
        l2C(:) = L2_error_cont;

        [~,L2_error_contP,~,~,~,~,~,~,~,~] = sampleSingleSpeedMC(T,alpha,beta,MCruns,theta_fun,d0,kappa,sigma,dt,dt,plotBool,i,velocityShift,randeinzug,'continuous',true,'upwind',[2-1000*eps,2+1000*eps]);
        l2PC(:) = L2_error_contP;
    end
end
tid = toc()

figure
y2 = log2(abs(l2C  - l2));      % continuous vs. piecewise-constant
y3 = log2(abs(l2PC - l2P));     % continuous+proxy vs. piecewise-constant+proxy
hold on
h2 = plot(log2(dtU), y2, LineWidth=1.5);
h3 = plot(log2(dtU), y3, LineWidth=1.5, LineStyle="--");
x = log2(dtU);
idxFit = x >= -4 & x <= 0;
p2 = polyfit(x(idxFit), y2(idxFit), 1);
p3 = polyfit(x(idxFit), y3(idxFit), 1);
xfit = linspace(-4, 0, 100);
plot(xfit, polyval(p2,xfit), ':', 'Color', h2.Color, 'LineWidth', 0.8)
plot(xfit, polyval(p3,xfit), ':', 'Color', h3.Color, 'LineWidth', 0.8)
xlabel('$\log_2(|\Pi_n|)$', 'Interpreter', 'latex')
ylabel('$\log_2(\mathrm{error})$', 'Interpreter', 'latex')
title('Logarithmic error for decreasing the control grid cell length including boundary', ...
    'Interpreter', 'latex', 'FontName', 'Arial')
legend( ...
    '$H(u^{(n),*})-H(u^*)$', ...
    '$H(\bar{u}^{(n)})-H(\bar{u})$', ...
    ['$\mathrm{fit}: s=' sprintf('%.3f',p2(1)) '$'], ...
    ['$\mathrm{fit}: s=' sprintf('%.3f',p3(1)) '$'], ...
    'Interpreter', 'latex', 'Location', 'best')
hold off
