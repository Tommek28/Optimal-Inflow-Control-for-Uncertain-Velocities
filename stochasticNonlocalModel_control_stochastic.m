function [output] = stochasticNonlocalModel_control_stochastic(xrange,u,dx,dt,T,v,gamma,DD)
% Like stochasticNonlocalModel_control, but averaged over an entire
% ensemble v (cell array of speed functions, one per MC realization,
% simulated internally via parfor) and directly evaluated as the expected
% squared error to the demand DD - this is the actual stochastic
% objective for fmincon (see skriptNonLocal.m).
%
% gamma: 5-point Gauss kernel weights precomputed by
% computeNonlocalKernelWeights(dx,eta,W) (see skriptNonLocal.m). This way
% the quadrature does not need to be recomputed on every one of the
% potentially thousands of calls (e.g. inside fmincon).

%% Sizes
% +1: see stochasticNonlocalModel.m - the time grid must include t=T,
% otherwise outflow(j) is offset by one time step relative to D/DD.
Nt   = round(T/dt) + 1;
Nx   = round((xrange(2)-xrange(1))/dx);
Neta = length(gamma);

%% Preallocate memory
rho     = zeros(Nt,Nx);
outflow = zeros(Nt,1);
% round((T-3)/dt)+1 instead of Nt/T*(T-3)+1: the old formula was only an
% integer for T/dt combinations that exactly divide Nt, and since the +1
% above it is no longer consistent with the actual slice length below
% (end-round((T-3)/dt)):end anyway.
outflowN = zeros(round((T-3)/dt)+1,size(v,1));

gammaNorm = cell(Neta+1,1);

for cut = 0:Neta
    g = gamma(1:end-cut);
    gammaNorm{cut+1} = g/sum(g);
end

%% Initialization
rho(1,1) = u(1);

%% Helper quantities
interiorEnd = Nx - Neta;

%% MC
parfor q = 1:size(v,1)

    vfun = v{q};

    rho = zeros(Nt,Nx);
    outflow = zeros(Nt,1);

    rho(1,1) = u(1);

    for j = 2:Nt

        rho_old = rho(j-1,:);
        vrho = vfun(rho_old);

        inflowVel = sum(gamma .* vrho(1:Neta));
        rho(j,1) = u(j) / inflowVel;

        rho_new = rho_old;

        for i = 2:Nx

            if i >= interiorEnd

                idx = (Nx-Neta+1):Nx;
                vvP = sum(gamma .* vrho(idx));
                vvM = vvP;

            else

                cutP = max(0,i+Neta-Nx);
                gammaP = gammaNorm{cutP+1};

                idxP = i+1:i+Neta-cutP;
                vvP = sum(gammaP .* vrho(idxP));

                cutM = max(0,i-1+Neta-Nx);
                gammaM = gammaNorm{cutM+1};

                idxM = i:i-1+Neta-cutM;
                vvM = sum(gammaM .* vrho(idxM));

            end

            rho_new(i) = rho_old(i) ...
                - dt/dx * (rho_old(i)*vvP - rho_old(i-1)*vvM);

        end

        rho(j,2:end) = rho_new(2:end);
        outflow(j) = rho_new(end) * vrho(end);

    end

    % outflow(k) <-> t=(k-1)*dt, outflow(end) <-> t=T -> now correctly
    % aligned with DD = D(3/dt+1:end) (see stochasticNonlocalModel_control.m).
    outflowN(:,q) = outflow(end-round((T-3)/dt):end);

end

%output = dt*sum(mean(outflowN.^2,2) - 2*mean(outflowN,2).*DD + DD.^2);
output = dt*sum(mean((outflowN - DD).^2,2));
end
