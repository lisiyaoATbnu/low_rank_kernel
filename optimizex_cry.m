function [ x1, x2 ] = optimizex_cry( x1, x2, k, y1, y2, lambda, imax, jmax, t)
%OPTIMIZEX xi = argmin {\lambda/2 |xi*k - y|^2 + |x|_1/|x|_2}
%   x1 and y1 are x-derivatives of x and y; x2 and y2 are y-derivatives;
%   imax is the max iteration. t is the step of iteration.

% Using Iterative Shrinkage-Threshood Algorithms to optimize l1
% regularizations. 

% 
x1 = x1 * 6 / norm(x1(:));
x2 = x2 * 6 / norm(x2(:));

tmp1 = conv2(x1, k, 'same') - y1;
tmp2 = conv2(x2, k, 'same') - y2;
costLS0 = lambda / 2 * ((tmp1(:)' * tmp1(:) + tmp2(:)' * tmp2(:)));
costR0 =  norm(x1(:), 1) / norm(x1(:)) + norm(x2(:), 1) / norm(x2(:));
cost0 = costLS0 + costR0;

tp = 1e-4;

while t > tp
    x10 = x1;
    x20 = x2;
    for i = 1 : imax
%         fprintf('\tUpdating x, iteration %d... \n', i);
        l21 = norm(x1(:));
        l22 = norm(x2(:));
        for j = 1 : jmax
            % Here the gradient is calculated by method of Perone 2014 CVPR.
            gradient1 = lambda * conv2(conv2(x1, k, 'same') - y1,rot90(k, 2), 'same');
            gradient2 = lambda * conv2(conv2(x2, k, 'same') - y2,rot90(k, 2), 'same');

            % Based on Krishnan 2011 CVPR, regard l2  known by last iteration.
            tmp1 = x1 - t * l21 * gradient1;
            tmp2 = x2 - t * l22 * gradient2;
            s1 = sign(tmp1);
            s2 = sign(tmp2);
            x1 = max(0, abs(tmp1) -  t) .* s1;
            x2 = max(0, abs(tmp2) -  t) .* s2;
    
        end;
    end;
    
    tmp1 = conv2(x1, k, 'same') - y1;
    tmp2 = conv2(x2, k, 'same') - y2;
    costLS1 = lambda / 2 * (tmp1(:)' * tmp1(:) + tmp2(:)' * tmp2(:));
    costR1 =  ( norm(x1(:), 1) / norm(x1(:)) + norm(x2(:), 1) / norm(x2(:)) );
    cost1 = costLS1 + costR1;
    
    if cost1 > 2 * cost0 || isnan(cost1)
        t = t / 2;
        x1 = x10; 
        x2 = x20;
    else
        break;
    end;
end;


end

