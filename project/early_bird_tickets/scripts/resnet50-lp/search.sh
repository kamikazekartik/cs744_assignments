CUDA_VISIBLE_DEVICES=0 python eb_search_lp.py \
--eb_percent_prune 30 \
--dataset cifar10 \
--num_classes 10 \
--arch resnet50 \
--depth 50 \
--save ./lp_baseline/resnet50-cifar10-8b \
--lr 0.1 \
--epochs 160 \
--schedule 30 60 \
--batch-size 128 \
--test-batch-size 128 \
--momentum 0.9 \
--sparsity-regularization \
--swa True \
--swa_start 140 \
--wl-weight 8 \
--wl-grad 8 \
--wl-activate 8 \
--wl-error 8 \
--wl-momentum 8 \
--rounding stochastic
