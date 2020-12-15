python3 main.py --seed 42 \
--max-epochs 161 \
--lr 0.01 \
--gamma 0.998 \
--dataset Cifar10 \
--model preresnet \
--batch-size 256 \
--test-batch-size 1000 \
--use-amp False \
--use-half False \
--preload-data False \
--device=cuda \
--low-prec=True \
--nbits-weight=8 \
--nbits-activate=8 \
--nbits-grad=8 \
--nbits-error=8 \
--nbits-momentum=8 \
--results-file=output/v100_fullmodel_2/results_preresnet_Cifar10_batchsize256_precision8bit.csv \
> output/v100_fullmodel_2/log_preresnet_Cifar10_batchsize256_precision8bit 2>&1
