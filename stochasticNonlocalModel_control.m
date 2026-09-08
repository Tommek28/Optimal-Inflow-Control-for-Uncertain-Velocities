function [outflowN] = stochasticNonlocalModel_control(xrange,u,dx,dt,T,v,gamma)
% Like stochasticNonlocalModel, but returns only the time window relevant
% to the objective, outflowN (t in [3,T]) - for use as an objective
% function inside fmincon (see skriptNonLocal.m).
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

%% Initialization
rho(1,1) = u(1);

%% Helper quantities
interiorEnd = Nx - Neta;

%% Time loop
for j = 2:Nt

    rho_old = rho(j-1,:);

    % Evaluate speed only once
    vrho = v(rho_old);

    %% Inflow boundary condition

    inflowVel = sum(gamma .* vrho(1:Neta));

    rho(j,1) = u(j) / inflowVel;

    if rho(j,1) > 1
        disp('Something is wrong here')
        disp(rho(j,1))
    end

    %% Interior points

    for i = 2:Nx

        if i >= interiorEnd

            idx = (Nx-Neta+1):Nx;

            vvP = sum(gamma .* vrho(idx));
            vvM = vvP;

        else

            cutP = max(0,i+Neta-Nx);
            idxP = i+1:i+Neta-cutP;

            gammaP = gamma(1:end-cutP);
            gammaP = gammaP / sum(gammaP);

            vvP = sum(gammaP .* vrho(idxP));

            cutM = max(0,i-1+Neta-Nx);
            idxM = i:i-1+Neta-cutM;

            gammaM = gamma(1:end-cutM);
            gammaM = gammaM / sum(gammaM);

            vvM = sum(gammaM .* vrho(idxM));

        end

        rho(j,i) = rho_old(i) ...
                 - dt/dx * ...
                 ( rho_old(i)   * vvP ...
                 - rho_old(i-1) * vvM );

    end

    %% Outflow

    outflow(j) = rho(j,end) * v(rho(j,end));

end
    % Thanks to Nt=round(T/dt)+1, outflow(k) <-> t=(k-1)*dt now holds,
    % exactly like D(k) <-> t=(k-1)*dt in skriptNonLocal.m. outflow(end)
    % <-> t=T, so this window is now correctly aligned with
    % DD = D(3/dt+1:end).
    outflowN = outflow((end-round((T-3)/dt)):end);
end
