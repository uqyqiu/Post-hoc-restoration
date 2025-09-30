#! /bin/bash

# Multi-class
# retrain
python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 10

# boundary_shrink
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 10

# boundary_expanding
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 10

# unrolling
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 10

# unrolling_f
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 10

# salun
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 10

# ga
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 10

# BadT
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 10

# scrub
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 10

# Single-class
# retrain
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 1

# boundary_shrink
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 1

# boundary_expanding
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 1

# unrolling
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 1

# unrolling_f
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 1

# salun 
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 1

# ga
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 1

# BadT
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 1

# scrub
# python3 ../train_unlearn.py  --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar10 --model ResNet18 --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 1
# python3 ../train_unlearn.py  --dataset cifar100 --model ResNet34 --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 1





