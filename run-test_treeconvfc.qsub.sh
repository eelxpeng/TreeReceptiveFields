#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client110

CUDA_VISIBLE_DEVICES=1 python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset agnews --lr 0.001 --name agnews-treeconvfc |& tee logs/log-agnews-treeconvfc-layer-3-run-4
CUDA_VISIBLE_DEVICES=1 python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc |& tee logs/log-dbpedia-treeconvfc-layer-3-run-4
CUDA_VISIBLE_DEVICES=1 python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc |& tee logs/log-yelp-treeconvfc-layer-3-run-4
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-3
CUDA_VISIBLE_DEVICES=1 python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --name sogounews-treeconvfc |& tee logs/log-sogounews-treeconvfc-layer-3-run-4
