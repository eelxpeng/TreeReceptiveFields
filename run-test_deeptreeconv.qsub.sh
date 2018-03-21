#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_deeptreeconv.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-deep