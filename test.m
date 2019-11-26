clear;
close all;
clc;


%% Parameters
% roma large:
params.tx = 1e-2;
params.tau = 1e-5;

params.delta = 1e-5;
% total x-k iters
params.imax = 5;

%x step outer iter #
params.ximax = 2;

% x step inner iter #
params.xjmax = 2;

% k step kernel  iter #
params.kmax = 3;

%k step  low rank step #
params.rmax = 3 ;

% flag whether using low rank
params.sigma = 1;

% image reg param
params.lambda = 80; % or 80

% kernel threshold (threshold * max)
params.threshold = 0.05;

params.mu =1;

% k step total iter #
params.iterkrank = 10;

% kernel size
K = 185;

% whether show inter results (suggested true to tune...)
params.verbose = true;

% file name
fn = 'roma_large.jpg';

y = imread(fn);
y = im2double(y);

imshow(y);
pause;

 yori = y;
% y = yp

if size(y, 3) == 3
    yc = rgb2ycbcr(y);
    y = yc(:, : ,1);
else
    yc = y;
end;

  [x, k] = multiscaled_cry(y, K, params);    

if size(yc, 3) == 3
    yc(:, :, 1) = x;
    x = ycbcr2rgb(yc);
end;

y = yori;

imshow(x);
figure;
imagesc(k)

save('tmp_face2_35.mat', 'x', 'k', 'params');