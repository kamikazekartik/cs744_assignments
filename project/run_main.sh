python3 main.py --seed 42 \
--lr 0.01 \
--gamma 0.998 \
--dataset Cifar10 \
--model resnet50 \
--batch-size 128 \
--test-batch-size 1000 \
--use-amp True \
--use-half False \
--preload-data False \
--device=cuda \
--low-prec=False \
--nbits-weight=8 \
--nbits-activate=8 \
--nbits-grad=8 \
--nbits-error=8 \
--nbits-momentum=8 \
> cifar10_resnet50_amp_log 2>&1
