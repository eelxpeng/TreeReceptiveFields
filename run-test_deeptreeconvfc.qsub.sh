#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_deeptreeconvfc.py --batch-size 256 --epochs 20 --lr 0.001 --name agnews-deepfc