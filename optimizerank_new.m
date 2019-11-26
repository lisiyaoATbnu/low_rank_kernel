function [ k ] = optimizerank_new( k0,  imax, tau, delta)
%OPTIMIZEKPROX k = argmin 1/2\tau * |k - k0|^2 +  rank(k) 
%   The proxy of rank is logdet((k*k')^1/2 + \delta * I) = \Sum(log(sigular
%   value + \delta) with the first Taylor series extension.
%   k(i + 1) = U(X)(\Sigma(X) - \sigma * diag(w(k(i))))V'(X) 

% fprintf('\tOptimizing rank...\n');

% imax is experimentally < 10

% imagesc(k0);
% pause;

mu = 1;
% N = 20;
% for small size
% tau = 3e-4; 
%for large size
% tau = 1e-5;
X = k0;
% delta = 1e-5;
% imagesc(X);
w = mu * ones(size(X, 1), 1);
% pause;
% tmpL = X;
for i = 1 : imax
    
    [U S V]  = svd(X);
    L = U * max(S - tau * diag(w), 0) * V';
%     imagesc(L);
%     pause; 
% L(L<0) = 0;
% if sum(L(:))
% L = L/sum(L(:));
% end;
    w = 1./(svd(L) + delta);
%     tmpL  = L;



end;

k = L;

