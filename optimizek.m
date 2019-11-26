function [ k ] = optimizek( x1, x2, k, y1, y2, imax, mu)
%OPTIMIZEK k = argmin { |x * k - y|^2 + \sigma rank(k) }
% Here we use proxy to rank logdet(k + \delta I ) to optimize k;
% logdet(k + \delta I) ~= log(k_t + \delta I) + tr[(k_t + \delta I)^-1(k -
% k_t)] 
% 
% grad ~ (k_t + \delta I)^-T .* k
% Because this approximation is linear the whole loss function is convex.
% Using conjugate gradient and .

ep = 1e-3;

i = 0;
k0 = k;
% A = X'X
gradLS1 = 2 * conv2(rot90(x1, 2), conv2(x1, k, 'valid'), 'valid');
gradLS2 = 2 * conv2(rot90(x2, 2), conv2(x2, k, 'valid'), 'valid');
Ak = gradLS1 + gradLS2 + 2 * mu * k;

b = 2 * conv2(rot90(x1, 2), y1, 'valid') + 2 * conv2(rot90(x2, 2), y2, 'valid');
b = b + 2 * mu * k0;
% residule r(i) = -Ae(i)
r = b - Ak;

d = r;
e1 = r(:)' * r(:);
e0 = e1;

while e1 > ep * e0 && i < imax
%     fprintf('\tUpdating k, iteration %d ...\n', i);
    % calculate Ad
    gradLS1 = 2 * conv2(rot90(x1, 2), conv2(x1, d, 'valid'), 'valid');
    gradLS2 = 2 * conv2(rot90(x2, 2), conv2(x2, d, 'valid'), 'valid');
    Ad = gradLS1 + gradLS2 + 2 * mu * d; 
    q = Ad;
    alpha = e1 / (d(:)' * q(:));
    k = k + alpha * d;
       
    % to avoid round-off errors
    if ~mod(i, 50)
        
        gradLS1 = 2 * conv2(rot90(x1, 2), conv2(x1, k, 'valid') - y1, 'valid');
        gradLS2 = 2 * conv2(rot90(x2, 2), conv2(x2, k, 'valid') - y2, 'valid');
        gradLS = gradLS1 + gradLS2 + 2 * mu * (k - k0);

        r = -gradLS;
    else
        r = r - alpha * q;
      
    end;
    
    e0 = e1;
    e1 = r(:)' * r(:);
    beta = e1 / e0;
    d = r + beta * d;
    i = i + 1;
    
end;


end

