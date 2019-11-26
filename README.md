# low_rank_kernel
Code implementation of paper "Understanding kernel size in blind deconvolution", which is accepted by WACV 2019.

In this paper, we analyzed why oversized kernel will lower the quality of deblurred images in conventional blind deconvolution methods. We proposed a low-rank based method to suppress this effect. 

The non-blind deconvolution method used in this paper is proposed in "Fast Image Deconvolution using Hyper-Laplacian Priors" by Krishnan et al. The files"center_kernel_seperate.m", "fast_deconv_bregman.m", "solve_image_bregman" and the settings of non-blind deconvolution are forked from the author's website.

This code is only permitted for non-commercial usage. Please cite our paper if you use this code.

