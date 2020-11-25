python main.py --seed 42 \
--lr 0.01 \
--gamma 0.998 \
--dataset MNIST \
--model lenet \
--batch-size 64 \
--test-batch-siz e 1000 \
--use-amp False \
--use-half False \
--preload-data False \
--device=cuda \
> mnist_lenet_full_prec_3epoch_log 2>&1
