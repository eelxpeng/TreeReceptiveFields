#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client114

python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset agnews --lr 0.001 --name agnews-treeconvfc > logs/log-agnews-treeconvfc-layer-3-run-4
python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc > logs/log-dbpedia-treeconvfc-layer-3-run-4
python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc > logs/log-yelp-treeconvfc-layer-3-run-4
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-3
python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --name sogounews-treeconvfc > logs/log-sogounews-treeconvfc-layer-3-run-4
