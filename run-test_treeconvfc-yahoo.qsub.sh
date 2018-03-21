#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

# python test_treeconvfc.py --batch-size 256 --epochs 50 --lr 0.001 --name agnews-treeconvfc
# python test_treeconvfc.py --batch-size 256 --epochs 20 --dataset dbpedia --lr 0.001 --name dbpedia-treeconvfc-2
# python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconvfc
python test_treeconvfc.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconvfc-7
