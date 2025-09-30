#! /bin/bash

python ../prepare_data.py --dataset mnist --trials 3 --seed 7 --request_rate 0.1 --val_rate 0.1
python ../prepare_data.py --dataset mnistFashion --trials 3 --seed 7 --request_rate 0.1 --val_rate 0.1
python ../prepare_data.py --dataset cifar10 --trials 3 --seed 7 --request_rate 0.1 --val_rate 0.1
python ../prepare_data.py --dataset cifar100 --trials 3 --seed 7 --request_rate 0.1 --val_rate 0.1