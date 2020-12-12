

PRECISION_LEVELS=('4' '6' '8')
PRUNING_LEVELS=(30 50)

for PRECISION_LEVEL in "${PRECISION_LEVELS[@]}"
do
	for PRUNING_LEVEL in "${PRUNING_LEVELS[@]}"
	do	
		python3 main.py --seed 42 \
		--lr 0.01 \
		--gamma 0.998 \
		--max-epochs 151 \
		--dataset Cifar10 \
		--num-classes 10 \
		--model vgg_pruned \
		--batch-size 64 \
		--test-batch-size 1000 \
		--use-amp False \
		--use-half False \
		--preload-data False \
		--device=cuda \
		--low-prec=True \
		--nbits-weight=$PRECISION_LEVEL \
		--nbits-activate=$PRECISION_LEVEL \
		--nbits-grad=$PRECISION_LEVEL \
		--nbits-error=$PRECISION_LEVEL \
		--nbits-momentum=$PRECISION_LEVEL \
		--pruned-model-path ~/cs744_assignments/project/Early-Bird-Tickets/lp_baseline/vgg16-cifar10-8b/pruned_${PRUNING_LEVEL}/pruned.pth.tar \
		--results-file=./output/results_vgg16_cifar10_${PRUNING_LEVEL}_${PRECISION_LEVEL}b_run \
		> ./output/log_vgg16_cifar10_${PRUNING_LEVEL}_${PRECISION_LEVEL}b_run 2>&1
	done
done
