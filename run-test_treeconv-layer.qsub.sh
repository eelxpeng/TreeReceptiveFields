#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client114

# Agnews
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconv-layer-3 > logs/log-agnews-treeconv-layer-3-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 1 --name agnews-treeconv-layer-1 > logs/log-agnews-treeconv-layer-1-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 2 --name agnews-treeconv-layer-2 > logs/log-agnews-treeconv-layer-2-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 4 --name agnews-treeconv-layer-4 > logs/log-agnews-treeconv-layer-4-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 5 --name agnews-treeconv-layer-5 > logs/log-agnews-treeconv-layer-5-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 6 --name agnews-treeconv-layer-6 > logs/log-agnews-treeconv-layer-6-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 7 --name agnews-treeconv-layer-7 > logs/log-agnews-treeconv-layer-7-run-1

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconv-layer-3 > logs/log-agnews-treeconv-layer-3-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 1 --name agnews-treeconv-layer-1 > logs/log-agnews-treeconv-layer-1-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 2 --name agnews-treeconv-layer-2 > logs/log-agnews-treeconv-layer-2-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 4 --name agnews-treeconv-layer-4 > logs/log-agnews-treeconv-layer-4-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 5 --name agnews-treeconv-layer-5 > logs/log-agnews-treeconv-layer-5-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 6 --name agnews-treeconv-layer-6 > logs/log-agnews-treeconv-layer-6-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 7 --name agnews-treeconv-layer-7 > logs/log-agnews-treeconv-layer-7-run-2

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 3 --name agnews-treeconv-layer-3 > logs/log-agnews-treeconv-layer-3-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 1 --name agnews-treeconv-layer-1 > logs/log-agnews-treeconv-layer-1-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 2 --name agnews-treeconv-layer-2 > logs/log-agnews-treeconv-layer-2-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 4 --name agnews-treeconv-layer-4 > logs/log-agnews-treeconv-layer-4-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 5 --name agnews-treeconv-layer-5 > logs/log-agnews-treeconv-layer-5-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 6 --name agnews-treeconv-layer-6 > logs/log-agnews-treeconv-layer-6-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --lr 0.001 --num-layer 7 --name agnews-treeconv-layer-7 > logs/log-agnews-treeconv-layer-7-run-3

# DBPedia
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 3 --name dbpedia-treeconv-layer-3 > logs/log-dbpedia-treeconv-layer-3-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 1 --name dbpedia-treeconv-layer-1 > logs/log-dbpedia-treeconv-layer-1-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 2 --name dbpedia-treeconv-layer-2 > logs/log-dbpedia-treeconv-layer-2-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 4 --name dbpedia-treeconv-layer-4 > logs/log-dbpedia-treeconv-layer-4-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 5 --name dbpedia-treeconv-layer-5 > logs/log-dbpedia-treeconv-layer-5-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 6 --name dbpedia-treeconv-layer-6 > logs/log-dbpedia-treeconv-layer-6-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 7 --name dbpedia-treeconv-layer-7 > logs/log-dbpedia-treeconv-layer-7-run-1

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 3 --name dbpedia-treeconv-layer-3 > logs/log-dbpedia-treeconv-layer-3-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 1 --name dbpedia-treeconv-layer-1 > logs/log-dbpedia-treeconv-layer-1-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 2 --name dbpedia-treeconv-layer-2 > logs/log-dbpedia-treeconv-layer-2-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 4 --name dbpedia-treeconv-layer-4 > logs/log-dbpedia-treeconv-layer-4-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 5 --name dbpedia-treeconv-layer-5 > logs/log-dbpedia-treeconv-layer-5-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 6 --name dbpedia-treeconv-layer-6 > logs/log-dbpedia-treeconv-layer-6-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 7 --name dbpedia-treeconv-layer-7 > logs/log-dbpedia-treeconv-layer-7-run-2

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 3 --name dbpedia-treeconv-layer-3 > logs/log-dbpedia-treeconv-layer-3-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 1 --name dbpedia-treeconv-layer-1 > logs/log-dbpedia-treeconv-layer-1-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 2 --name dbpedia-treeconv-layer-2 > logs/log-dbpedia-treeconv-layer-2-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 4 --name dbpedia-treeconv-layer-4 > logs/log-dbpedia-treeconv-layer-4-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 5 --name dbpedia-treeconv-layer-5 > logs/log-dbpedia-treeconv-layer-5-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 6 --name dbpedia-treeconv-layer-6 > logs/log-dbpedia-treeconv-layer-6-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --num-layer 7 --name dbpedia-treeconv-layer-7 > logs/log-dbpedia-treeconv-layer-7-run-3

# Yelp
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconv-layer-3 > logs/log-yelp-treeconv-layer-3-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconv-layer-1 > logs/log-yelp-treeconv-layer-1-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconv-layer-2 > logs/log-yelp-treeconv-layer-2-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconv-layer-4 > logs/log-yelp-treeconv-layer-4-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconv-layer-5 > logs/log-yelp-treeconv-layer-5-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconv-layer-6 > logs/log-yelp-treeconv-layer-6-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconv-layer-7 > logs/log-yelp-treeconv-layer-7-run-1

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconv-layer-3 > logs/log-yelp-treeconv-layer-3-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconv-layer-1 > logs/log-yelp-treeconv-layer-1-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconv-layer-2 > logs/log-yelp-treeconv-layer-2-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconv-layer-4 > logs/log-yelp-treeconv-layer-4-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconv-layer-5 > logs/log-yelp-treeconv-layer-5-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconv-layer-6 > logs/log-yelp-treeconv-layer-6-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconv-layer-7 > logs/log-yelp-treeconv-layer-7-run-2

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 3 --name yelp-treeconv-layer-3 > logs/log-yelp-treeconv-layer-3-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 1 --name yelp-treeconv-layer-1 > logs/log-yelp-treeconv-layer-1-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 2 --name yelp-treeconv-layer-2 > logs/log-yelp-treeconv-layer-2-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 4 --name yelp-treeconv-layer-4 > logs/log-yelp-treeconv-layer-4-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 5 --name yelp-treeconv-layer-5 > logs/log-yelp-treeconv-layer-5-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 6 --name yelp-treeconv-layer-6 > logs/log-yelp-treeconv-layer-6-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --num-layer 7 --name yelp-treeconv-layer-7 > logs/log-yelp-treeconv-layer-7-run-3

# Sogounews
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconv-layer-3 > logs/log-sogounews-treeconv-layer-3-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconv-layer-1 > logs/log-sogounews-treeconv-layer-1-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconv-layer-2 > logs/log-sogounews-treeconv-layer-2-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconv-layer-4 > logs/log-sogounews-treeconv-layer-4-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconv-layer-5 > logs/log-sogounews-treeconv-layer-5-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconv-layer-6 > logs/log-sogounews-treeconv-layer-6-run-1
python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconv-layer-7 > logs/log-sogounews-treeconv-layer-7-run-1

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconv-layer-3 > logs/log-sogounews-treeconv-layer-3-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconv-layer-1 > logs/log-sogounews-treeconv-layer-1-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconv-layer-2 > logs/log-sogounews-treeconv-layer-2-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconv-layer-4 > logs/log-sogounews-treeconv-layer-4-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconv-layer-5 > logs/log-sogounews-treeconv-layer-5-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconv-layer-6 > logs/log-sogounews-treeconv-layer-6-run-2
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconv-layer-7 > logs/log-sogounews-treeconv-layer-7-run-2

# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 3 --name sogounews-treeconv-layer-3 > logs/log-sogounews-treeconv-layer-3-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 1 --name sogounews-treeconv-layer-1 > logs/log-sogounews-treeconv-layer-1-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 2 --name sogounews-treeconv-layer-2 > logs/log-sogounews-treeconv-layer-2-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 4 --name sogounews-treeconv-layer-4 > logs/log-sogounews-treeconv-layer-4-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 5 --name sogounews-treeconv-layer-5 > logs/log-sogounews-treeconv-layer-5-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 6 --name sogounews-treeconv-layer-6 > logs/log-sogounews-treeconv-layer-6-run-3
# python test_treeconv-layer.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --num-layer 7 --name sogounews-treeconv-layer-7 > logs/log-sogounews-treeconv-layer-7-run-3


# python test_treeconv.py --batch-size 256 --epochs 50 --dataset dbpedia --lr 0.001 --name dbpedia-treeconv
# python test_treeconv.py --batch-size 256 --epochs 50 --dataset yelp --lr 0.001 --name yelp-treeconv
# python test_treeconv.py --batch-size 256 --epochs 50 --dataset yahoo --lr 0.001 --name yahoo-treeconv-3
# python test_treeconv.py --batch-size 256 --epochs 50 --dataset sogounews --lr 0.001 --name sogounews-treeconv
