
python3 main.py --seed 42 \
--lr 0.01 \
--gamma 0.998 \
--dataset Cifar10 \
--num-classes 10 \
--model preresnet \
--batch-size 64 \
--test-batch-size 1000 \
--use-amp False \
--use-half False \
--preload-data False \
--device=cuda \
--low-prec=False \
--nbits-weight=8 \
--nbits-activate=8 \
--nbits-grad=8 \
--nbits-error=8 \
--nbits-momentum=8 \
--pruned-model-path ./early_bird_tickets/baseline/resnet-nolp-cifar10/pruned_50/pruned.pth.tar
> ./output/preresnet_log 2>&1

