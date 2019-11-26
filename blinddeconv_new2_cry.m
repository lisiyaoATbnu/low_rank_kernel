function [ x1 x2 k ] = blinddeconv_new2_cry( y1, y2, x1, x2, lambda, sigma, k, imax, ximax, xjmax, kimax, rimax, iterkrank, tx, mu, tau, delta, threshhold, L2norm, verbose )
%BLINDDCONV deblurs the given picture y.
%   (x, k) = argmin_(x, k) { |x*k-y|^2 + \lambda|x1|_1/|x|_2 + \sigma
%   rank(k) }.

%  \delta is assumed to be small used in the heuristic proxy of rank:
%  rank(x) ~ logdet(x + \delta I). ximax and kimax is the maximum iteration
%  number of optimizations on x's and k's.

% As a very important hyper-parameter, K is assumed to be larger then
% ground truth to test the effeciency of low rank regularization. K MUST be
% odd and >= 3.

%% Kernel estimation (blind deconvolution)

% initializing

if sum(abs(x1(:))) == 0
    x1 = y1;
end;
if sum(abs(x2(:))) == 0
    x2 = y2;
end;

% if sum(abs(x1(:)) > 0) < sum(abs(y1(:) > 0)) / 20
%     x1 = y1;
% end;
% if sum(abs(x2(:)) > 0) < sum(abs(y2(:) > 0)) / 20
%     x2 = y2;
% end;

x1 = x1 * L2norm  / norm(x1(:));
x2 = x2 * L2norm / norm(x2(:));

ksz = size(k, 1);
bhs = floor(ksz / 2);
y1v = y1(bhs + 1 : end - bhs, bhs + 1 : end - bhs);
y2v = y2(bhs + 1 : end - bhs, bhs + 1 : end - bhs);

changes = zeros(imax, 1);
k0 = k;
for i = 1 : imax
    fprintf('Iteration %d...\n', i);
    
        if verbose ==  true
%         subplot(2, 2, 1);
%         
%         imagesc(x1);
%         colorbar
%         subplot(2, 2, 2);
%         imagesc(x2);
%         colorbar
%          subplot(2, 2, 3);
%         imagesc(k);
%         colorbar
%         
%         subplot(2, 2, 4);
%         imshow(y1)
%         colormap('jet')
%         drawnow
imagesc(k);
colormap jet
axis off
drawnow
        end
    
    % updating x
    [x1, x2] = optimizex_cry(x1, x2, k, y1, y2, lambda, ximax, xjmax, tx);
    for iter = 1 : iterkrank
        if iter == 1
            tmpmu = 0;
        else
            tmpmu = mu * exp(iter) / exp(iterkrank);
        end;
        k = optimizek(x1, x2, k, y1v, y2v, kimax, tmpmu);
        if (sigma > 0)
            k = optimizerank_new( k, rimax, tau, delta);
        end;
        
        k(k < 0) = 0;
        k = k / sum(k(:));
    end
 
    %     projection to nonegative and sum 1
    if threshhold
        k(k < max(max(k)) * threshhold * i/imax ) = 0;
    else
        k( k < 0 ) = 0;
    end;
    k = k / sum(k(:));
    
end;