#!/bin/bash

model_data=( 1 )
baseline_dir="./Early-Bird-Tickets/lp_baseline"
echo "Baseline Dir:"
ls $baseline_dir


model_paths=( \
	${baseline_dir}/vgg11-cifar10-8b/pruned_30_0.3/pruned.pth.tar \
	${baseline_dir}/vgg11-cifar10-8b/pruned_50_0.5/pruned.pth.tar \
	${baseline_dir}/vgg16-cifar10-8b/pruned_30_0.3/pruned.pth.tar \
	${baseline_dir}/vgg16-cifar10-8b/pruned_50_0.5/pruned.pth.tar
)
models=( \
	"vgg11" \
	"vgg11" \
	"vgg16" \
	"vgg16" \
)
prune_pcts=( \
	30 \
	50 \
	30 \
	50 \
)

for i in "${!model_paths[@]}"
do
	mp=${model_paths[$i]}
	model_name=${models[$i]}
	prune_pct=${prune_pcts[$i]}
	model=vgg_pruned
	data=Cifar10
	batch_size=( 128 )
	for bs in "${batch_size[@]}"
	do
		precision=( 1 2 3 )
		for pr in "${precision[@]}"
		do
			if [ $pr -eq 1 ]; then
				prec=Full
				amp=False
				half=False
			elif [ $pr -eq 2 ]; then
				prec=AMP
				amp=True
				half=False
			else
				prec=Half
				amp=False
				half=True
			fi
			echo "Processing: Model=$model; Data=$data; Batch Size=$bs; AMP=$amp; Half=$half; Model_Path=$mp; Model Name=$model_name; Prune Percent=$prune_pct"
			python3 main.py --seed 42 \
				--max-epochs 120 \
				--lr 0.01 \
				--gamma 0.998 \
				--dataset $data \
				--model $model \
				--pruned-model-path $mp \
				--num-classes 10 \
				--batch-size $bs \
				--test-batch-size 1000 \
				--use-amp $amp \
				--use-half $half \
				--preload-data False \
				--device=cuda \
				--low-prec=False \
				--nbits-weight=8 \
				--nbits-activate=8 \
				--nbits-grad=8 \
				--nbits-error=8 \
				--nbits-momentum=8 \
				--results-file=output/results_${model_name}_${prune_pct}_${data}_batchsize${bs}_${prec} \
				> output/log_${model_name}_${prune_pct}_${data}_batchsize${bs}_${prec} 2>&1

		done
	done
done
