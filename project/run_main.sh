python3 main.py --seed 42 \
--max-epochs 150 \
--lr 0.1 \
--gamma 0.998 \
--dataset Cifar10 \
--model resnet50 \
--batch-size 128 \
--test-batch-size 1000 \
--use-amp False \
--use-half True \
--preload-data False \
--device=cuda \
--low-prec=False \
--nbits-weight=8 \
--nbits-activate=8 \
--nbits-grad=8 \
--nbits-error=8 \
--nbits-momentum=8 \
> log_convergence_resnet50_Cifar10_batchsize128_Half 2>&1
