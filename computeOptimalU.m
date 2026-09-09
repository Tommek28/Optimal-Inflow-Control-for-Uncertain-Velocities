function u_vals = computeOptimalU(t_grid, d0, kappa, theta_fun, dt_int, velocityShift, alpha, beta, applyBoundary, outputMode)
% Optimal inflow control, per Theorem 3.1 (continuous-time) /
% Proposition 3.3 (piecewise-constant) of the paper. 
% Considering two independent switches:
%   applyBoundary (logical): true restricts the averaging weight to the
%     admissible arrival-time set Lambda(t) from (3.2), i.e. the
%     boundary-corrected control u*(t) (Theorem 3.1); false uses the
%     "unconditioned interior expression" E_lambda[m(t+1/lambda)]
%     without that restriction (used for the Figure 6 boundary
%     comparison).
%   outputMode ('piecewise'|'continuous'): 'piecewise' aggregates the
%     fine-grid solution onto the cells of t_grid (piecewise-constant
%     control, Proposition 3.3); 'continuous' returns the fine-grid
%     solution directly (continuous-time control, Theorem 3.1).
%
% velocityShift, alpha, beta jointly define the assumed law of lambda: a
% Beta(alpha,beta) distribution on [0,1], affinely shifted and stretched
% onto [velocityShift(1),velocityShift(2)] - i.e. the same distribution
% that lambda is sampled from in sampleSingleSpeedMC.m via
% betarnd(alpha,beta,...). alpha=beta=1 recovers the uniform distribution
% (Beta(1,1) = U(0,1)) as a special case. Note: for alpha<1 or beta<1 the
% density is unbounded at the corresponding edge of the support; dl below
% includes the exact edges, so such shape parameters can produce Inf/NaN
% in varphi and would need a strictly-interior quadrature grid instead.

m_fun = @(t) jacobiMeanClosed(t, d0, kappa, theta_fun);
dl = linspace(velocityShift(1),velocityShift(2),100);
% Density of lambda = velocityShift(1) + (velocityShift(2)-velocityShift(1))*Y,
% Y ~ Beta(alpha,beta) - see betarnd(...) in sampleSingleSpeedMC.m.
% betapdf is 0 outside [0,1], so l outside velocityShift is handled automatically.
varphi = @(l) betapdf((l-velocityShift(1))/(velocityShift(2)-velocityShift(1)), alpha, beta) ...
              / (velocityShift(2)-velocityShift(1));

% uStar(k): optimal control on the fine grid s, averaging m_fun over the
% admissible velocities dl (weighted by varphi and, if applyBoundary,
% additionally restricted to Lambda(t) - i.e. arrival times s(k)+1/dl
% that fall inside [1/velocityShift(1), t_grid(end)]).
s = t_grid(1):dt_int:t_grid(end);
uStar = zeros(1,length(s));
q = zeros(length(s),length(dl));
for k = 1:length(s)
    if applyBoundary
        q(k,:) = (varphi(dl).*((s(k)+1./dl)>=1/velocityShift(1)).*((s(k)+1./dl)<=t_grid(end)))';
    else
        q(k,:) = (varphi(dl))';
    end
    if sum(q(k,:)) ~= 0
        uStar(k) = sum(m_fun(s(k)+1./dl).*q(k,:))/sum(q(k,:));
    end
end

if strcmp(outputMode, 'continuous')
    u_vals = uStar;
    return
end

% Piecewise-constant control: q-weighted average of uStar over each
% control cell [t_grid(i), t_grid(i+1)) (Proposition 3.3).
N = length(t_grid) - 1;
u_vals = zeros(1, N);
for i = 1:N
    idx1 = round(t_grid(i)/dt_int) + 1;
    idx2 = round(t_grid(i+1)/dt_int) + 1;
    u_vals(i) = sum(sum(q(idx1:idx2,:),2)'.*uStar(idx1:idx2))/sum(sum(q(idx1:idx2,:),2));
end
end
