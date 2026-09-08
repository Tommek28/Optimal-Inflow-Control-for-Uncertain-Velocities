function m_t = jacobiMeanClosed(t_query, d0, kappa, theta_fun)
% Closed-form representation via numerical quadrature
%
% m(t) = e^{-kappa t} d0 + kappa * int_0^t e^{-kappa(t-s)} theta(s) ds
%      = e^{-kappa t} * (d0 + kappa * G(t)),   G(t) := int_0^t e^{kappa s} theta(s) ds
%
% G(t) is the expensive part (it requires integrating from 0 to t), so it
% is cached: computed once via cumulative trapezoidal integration on a
% fine grid, then evaluated at any t via interpolation, instead of
% re-integrating from scratch for every single query point as before.
% This matters because jacobiMeanClosed is called from inside a loop of
% up to ~30,000 iterations per computeOptimalU.m call (each iteration
% querying it again) - previously the dominant cost of the whole
% pipeline (~70s per computeOptimalU.m call at T=16; now a small fraction
% of a second, see the repository README).
%
% The cache is keyed on (kappa, d0, theta_fun) and automatically rebuilt
% - extending the covered time range with a margin - whenever those
% change or a query falls outside the currently cached range. Being a
% MATLAB `persistent` variable, each parallel worker (parfor) keeps its
% own independent cache; there is no shared/race-condition risk.

persistent cache_kappa cache_d0 cache_theta_fun cache_tmax cache_tgrid cache_G

Ns_per_unit = 4000;   % quadrature resolution of the cached table (nodes per unit time)
margin = 1.2;         % rebuild covers 20% beyond the currently requested range,
                       % so that a slowly growing query range (e.g. across the
                       % k-loop in computeOptimalU.m) does not trigger a full
                       % rebuild on every single call

t_max_needed = max(t_query(:));

needs_rebuild = isempty(cache_kappa) || cache_kappa ~= kappa || cache_d0 ~= d0 ...
    || ~isequal(cache_theta_fun, theta_fun) || t_max_needed > cache_tmax;

if needs_rebuild
    t_max = max(t_max_needed, 1) * margin;
    Ns = max(ceil(Ns_per_unit * t_max), 2);
    cache_tgrid = linspace(0, t_max, Ns);
    integrand = exp(kappa*cache_tgrid) .* theta_fun(cache_tgrid);
    cache_G = cumtrapz(cache_tgrid, integrand);
    cache_kappa = kappa;
    cache_d0 = d0;
    cache_theta_fun = theta_fun;
    cache_tmax = t_max;
end

G_t = interp1(cache_tgrid, cache_G, t_query, 'linear');
m_t = exp(-kappa*t_query) .* (d0 + kappa*G_t);
end
