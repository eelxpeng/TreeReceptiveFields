#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_treeconvfc.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-treeconvfc-5
# python test_treeconvfc.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-treeconvfc-kernel-6