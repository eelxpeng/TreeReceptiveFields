#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client110

# python test_treeconvfc-bin.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-treeconvfc
# python test_treeconvfc-bin.py --batch-size 256 --epochs 20 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc-bin-2
# python test_treeconvfc-bin.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc-bin
# python test_treeconvfc-bin.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-bin
# python test_treeconvfc-bin.py --batch-size 256 --epochs 20 --dataset sogounews --lr 0.001 --name sogounews-treeconvfc-bin

CUDA_VISIBLE_DEVICES=1 python test_treeconvfc-bin.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-bin |& tee logs/log-yahoo-treeconvfc-layer-3-run-4
