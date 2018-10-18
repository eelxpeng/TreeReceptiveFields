#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
# #$ -l h=client110

python test_treeconvfc-mnist.py --batch-size 256 --epochs 50 --dataset mnist --lr 0.001 --name mnist-treeconvfc-151 |& tee logs/log-mnist-treeconvfc-151
