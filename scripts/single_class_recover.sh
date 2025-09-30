# #!/bin/bash
# mnist
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_unlearn_classes 1
python3 ../recover.py --dataset mnist --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_unlearn_classes 1

# mnistFashion
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_unlearn_classes 1
python3 ../recover.py --dataset mnistFashion --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_unlearn_classes 1

# cifar10
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_unlearn_classes 1
python3 ../recover.py  --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_unlearn_classes 1
python3 ../recover.py --dataset cifar10 --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_unlearn_classes 1

# cifar100
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method retrain --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_shrink --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method boundary_expanding --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method embedding_shift --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unrolling_f --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method unsc --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method salun --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method ga --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method fisher --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method BadT --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method sparse --n_unlearn_classes 1
python3 ../recover.py --dataset cifar100 --model AllCNN --trials 3 --seed 7 --unlearn-method scrub --n_unlearn_classes 1
