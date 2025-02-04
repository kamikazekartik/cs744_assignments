CUDA_VISIBLE_DEVICES=0 python vggprune_lp.py \
--eb_percent_prune 50 \
--dataset cifar10 \
--num_classes 10 \
--test-batch-size 256 \
--depth 11 \
--percent 0.5 \
--model ./lp_baseline/vgg11-cifar10-8b/EB-50.pth.tar \
--save ./lp_baseline/vgg11-cifar10-8b/pruned_50_0.5 \
--wl-weight 8 \
--wl-grad 8 \
--wl-activate 8 \
--wl-error 8 \
--wl-momentum 8 \
--rounding stochastic
