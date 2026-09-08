function [outflow,rho] = transportUpwind(lambda, u, dt, T)
% Simulates rho_t + lambda rho_x = 0 with CFL=1 (upwind), per path.
%
% INPUT:
% lambda  - (N x 1) speeds
% u       - function handle @(t) or vector (1 x Nt)
% dt      - time step
% T       - final time
%
% OUTPUT:
% outflow - (N x Nt) outflow at x=1
% rho     - final delay-line buffer state of the last path (k=N); not
%           used by any caller in this repository, kept only for
%           interface compatibility with the previous implementation.
%
% Implementation note: at CFL=1, this upwind scheme for constant-speed
% linear transport is exactly a delay line - each time step, the whole
% grid shifts by one cell. The previous implementation realized this
% literally via rho(2:end)=rho(1:end-1), an O(Nx) array copy at every
% one of ~T/dt time steps for every Monte Carlo path (Nx up to ~2000)
% - the dominant cost of the whole simulation pipeline at realistic
% MCruns (see the repository README, "Performance"). It is now
% implemented via a circular buffer (O(1) per step) instead - the same
% finite-difference/upwind scheme, only the data structure changed.
%
% While rewriting this, a bug was found and fixed: the previous
% rho(1)=v; rho(2:end)=rho(1:end-1); ordering had MATLAB evaluate the
% right-hand side of the shift *after* rho(1) was already overwritten,
% duplicating the new value into two cells per step and shortening the
% effective delay by one time step relative to the intended CFL=1
% scheme (Nx-2 cells of actual delay instead of Nx-1). The circular
% buffer below reproduces the corrected (non-duplicating) delay of
% Nx-1 cells - the delay a clean "write new value, then read the cell
% that is about to be overwritten" upwind step gives at CFL=1. This
% changes existing results by at most 1*dt out of a physical delay of
% 1/lambda (a relative shift on the order of 1e-3 to 1e-4 given dt vs.
% the lambda ranges used in this repository).

t = 0:dt:T;
Nt = length(t);
N = length(lambda);

outflow = zeros(N, Nt);

for k = 1:N   % over all paths
    lam = lambda(k);

    dx = lam * dt;
    Nx = ceil(1 / dx);      % number of spatial cells
    dx = 1 / Nx;            % slightly adjusted

    bufLen = max(Nx - 1, 1);   % see header note on the Nx-1 delay
    buf = zeros(1, bufLen);
    pos = 1;

    for n = 1:Nt-1
        % Boundary value
        if isa(u, 'function_handle')
            u_val = u(t(n));
        else
            u_val = u(n);
        end

        rho_in = u_val / lam;

        % Guard against infinite / NaN
        if ~isfinite(rho_in)
            rho_in = 0;
        end

        % Read the cell about to be overwritten (its value was written
        % bufLen steps ago, i.e. this is the delay-line output), then
        % write the new inflow value into that same slot.
        outflow(k,n) = buf(pos) * lam;
        buf(pos) = rho_in;
        pos = pos + 1;
        if pos > bufLen
            pos = 1;
        end
    end

    % last time point: no new input is fed in at t=T, so this repeats
    % the previous outflow value, matching the original implementation
    outflow(k, Nt) = outflow(k, Nt-1);
end

rho = buf;
end
