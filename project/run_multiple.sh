#!/bin/bash

model_data=( 1 2 3 )
for md in "${model_data[@]}"
do
	if [ $md -eq 1 ]; then
		model=resnet50
		data=Cifar10
	elif [ $md -eq 2 ]; then
		model=vgg16
		data=Cifar10
	else
		model=lenet
		data=EMNIST
	fi
	batch_size=( 64 128 256 512 1024 )
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
			echo "Processing: Model=$model; Data=$data; Batch Size=$bs; AMP=$amp; Half=$half"
			python3 main.py --seed 42 \
				--max-epochs 6 \
				--lr 0.01 \
				--gamma 0.998 \
				--dataset $data \
				--model $model \
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
				--results-file=output/results_${model}_${data}_batchsize${bs}_${prec} \
				> output/log_${model}_${data}_batchsize${bs}_${prec} 2>&1

		done
	done
done
