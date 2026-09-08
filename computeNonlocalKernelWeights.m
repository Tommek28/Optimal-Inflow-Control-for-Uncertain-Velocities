function gamma = computeNonlocalKernelWeights(dx, eta, W)
% Precomputes the 5-point Gauss kernel weights for the
% stochasticNonlocalModel* function family. dx, eta and W never change in
% skriptNonLocal.m between the (potentially thousands, during fmincon)
% model calls - so gamma is computed here exactly once and passed on to
% the model functions, instead of re-evaluating the quadrature on every
% single call.

Neta = round(eta/dx);

xi = [-0.906179845938664;
      -0.538469310105683;
       0.0;
       0.538469310105683;
       0.906179845938664];

w  = [0.236926885056189;
      0.478628670499366;
      0.568888888888889;
      0.478628670499366;
      0.236926885056189];

xiShift = 0.5*dx*(xi + 1);

X = (0:Neta-1)'*dx + xiShift';
gamma = 12.5*dx*sum(w'.*W(X),2)';
end
