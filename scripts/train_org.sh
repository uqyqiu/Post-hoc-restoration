#! /bin/bash

python ../train_original.py --dataset mnist --model AllCNN --trials 3 --seed 7
python ../train_original.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7
python ../train_original.py --dataset cifar10 --model ResNet18 --trials 3 --seed 7
python ../train_original.py --dataset cifar100 --model ResNet34 --trials 3 --seed 7