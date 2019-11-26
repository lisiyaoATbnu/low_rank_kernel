function [ x, k ] = multiscaled_cry( y, K, params )
%MULTISCALE deblur image y to x with final kernel size K and other
%parameters
%   Input:
%   -y: the blurry image;
%   -K: the expected kernel size (MUST be odd);
%   -params:
%       --lambda: the trade-off used in updating x;
%       --sigma: whether to use regularization in updating k;
%       --imax: the max iteration # for each scale;
%       --imax, imaxj: the max iteration # of minimizing |x| based on new 2-norm and # of updating |x| during updating x;
%       --kmax: max iteration # to optimize k;
%       --rmax: max iteration # of low rank regularization
%       --tx, tau: step params used in optimizign x and k, respectively;
%       --delta: a little small number used in low rank reg (usally ~10^-5)
%   We follow Krishnan CVPR 2011 code to design layers of the scaling
%   pyramid (/16).
verbose = params.verbose;
assert(mod(K, 2) == 1);



% calculate each layer
minscale = max(2 * floor((K - 1) / 32) + 1, 3);
layer = minscale;
i = 1;
step = sqrt(2);

while layer < K
    scale(i) = layer;
    i = i + 1;
    layer = floor(layer * step);
    if (mod(layer , 2) == 0)
        layer = layer + 1;
    end
end

scale(end + 1) = K;

k = zeros(minscale, minscale);
k(ceil(minscale / 2), ceil(minscale / 2) - 1 : ceil(minscale / 2)) = 0.5;
% k = ones(minscale, minscale) / sum(minscale ^ 2);

x1 = zeros(minscale, minscale);
x2 = zeros(minscale, minscale);

for i = 1 : length(scale)
    Ki = scale(i);
    fprintf('Proccessing ksize =  %d\n', Ki);
    ratio = Ki / K;
    hw = floor(size(y) * ratio);
    smally = imresize(y, hw, 'bilinear');
    x1 = imresize(x1, hw - 1, 'bilinear');
    x2 = imresize(x2, hw - 1, 'bilinear');
    if i ~= 1
        k = imresize(k, [ Ki, Ki ], 'bilinear');
    end
    
    dx = [1, -1; 0, 0];
    dy = [1, 0; -1, 0];
    
    L2norm = 6 * Ki / scale(1);
    
    y1 = conv2(smally, dx, 'valid');
    y2 = conv2(smally, dy, 'valid');

    
    [x1, x2, k] = blinddeconv_new2_cry(y1, y2, x1, x2, params.lambda, params.sigma, k, params.imax, params.ximax, ...
        params.xjmax, params.kmax, params.rmax, params.iterkrank, params.tx, params.mu,  params.tau * i / length(scale), params.delta, params.threshold,  L2norm, verbose);
%     tmp = zeros(size(y));
  
      [y1 x1 k] = center_kernel_separate(y1, x1, k);
    [y2 x2 k] = center_kernel_separate(y2, x2, k);
    
end
    
k(k < max(k(:)) * params.threshold) = 0;
k = k / sum(k(:));

%   [x1 x1 k] = center_kernel_separate(x1, x1, k);
%     [x2 x2 k] = center_kernel_separate(x2, x2, k);

%% Non-blind deconvolution
bhs = floor(K / 2);
padsize = bhs;
nb_lambda = 3000;
nb_alpha = 0.5;
kernel = k;

use_ycbcr = 1

if (use_ycbcr)
    if size(y, 3) == 3
        ycbcr = rgb2ycbcr(y);
    else
        ycbcr = y;
    end;
    nb_alpha = 1;
end;

if use_ycbcr == 1
    
    ypad = padarray(ycbcr(:,:,1), [padsize padsize], 'replicate', 'both');
    for a = 1:4
        ypad = edgetaper(ypad, kernel);
    end;
    
    tmp = fast_deconv_bregman(ypad, kernel, nb_lambda, nb_alpha);
    x(:, :, 1) = tmp(bhs + 1 : end - bhs, bhs + 1 : end - bhs);
    if (size(ycbcr, 3) == 3)
        x(:, :, 2:3) = ycbcr(:, :, 2:3);
        x = ycbcr2rgb(x);
    end;
else
    for j = 1:3
        ypad = padarray(y(:, :, j), [1 1] * bhs, 'replicate', 'both');
    end;
    for a = 1:4
        ypad = edgetaper(ypad, kernel);
    end;
    
    tmp = fast_deconv_bregman(ypad, kernel, nb_lambda, nb_alpha);
    x(:, :, j) = tmp(bhs + 1 : end - bhs, bhs + 1 : end - bhs);
end;


end

