#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune False --label_strat gcd --attack _default_optimization_attack --balanced False

python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced False

python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced False

python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced False

python Leakage.py --config ./configs/leakage/wear_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune False --label_strat gcd --attack _default_optimization_attack --balanced True

python Leakage.py --config ./configs/leakage/wetlab_loso_deep.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced True

python Leakage.py --config ./configs/leakage/wear_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced True

python Leakage.py --config ./configs/leakage/wetlab_loso_tiny.yaml --seed 1 --eval_type loso --trained True --neptune True --label_strat gcd --attack _default_optimization_attack --balanced True
