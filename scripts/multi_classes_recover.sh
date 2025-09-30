# #!/bin/bash
# mnist
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2

# mnistFashion
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2

# cifar10
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 2
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_group 10 --n_unlearn_classes 2
python3 ../recover.py --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 2

# cifar100
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_group 10 --n_unlearn_classes 10
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_group 10 --n_unlearn_classes 10
