python vggprune.py \
--eb_percent_prune 30 \
--dataset cifar10 \
--num_classes 10 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16-cifar10/EB-30.pth.tar \
--save ./baseline/vgg16-cifar10/pruned_30_0.3 \
--gpu_ids 0
