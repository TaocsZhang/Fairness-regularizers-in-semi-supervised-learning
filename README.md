# Fairness-regularizers-in-semi-supervised-learning

Authors: **Tao Zhang**, Tianqing Zhu, Kun Gao, Wanlei Zhou

This is part of code for my paper 'Fairness in Graph-based Semi-supervised Learning'. 


## Introduction
Our approach,  fair graph neural networks (FGNN), is built with GNNs, where the loss function includes classification loss and fairness loss. Classification loss optimizes the classification accuracy over all labeled data, and fairness loss enforces fairness over labeled data and unlabeled data.  GNN  models  combine  graph  structures  and  features, and  our  method  allows  GNN  models  to  distribute  gradient information from the classification loss and fairness loss. Thus, fair representations of nodes with labeled and unlabeled datacan be learned to achieve the ideal trade-off between accuracy and fairness.

## Requirements
Python 3.6<br>
Pytorch 1.2<br>
Pandas<br>
Numpy<br>

## Getting started

Fairness regularizers and data pre-processing are given in the utils.py.

Graph neural network is defined in the model.py and layer.py.

The training process is in the train.py. The following is an example to execute train.py.

python train.py --lr=0.005 --fare=1 --fair_metric=1 --alpha=0.5 --num_unlabel=400 --num_labeled=1000 

## Datasets

Three dataset are used in this paper, and the links are given in the following.

Bank dataset: https://archive.ics.uci.edu/ml/datasets/bank+mar

Health dataset: https://foreverdata.org/1015/index.html

Titanic dataset: ttps://www.kaggle.com/c/titanic/data

## Evaluation 
Fairness metrics.py is used to evaluate discrimination level with demographic parity and equal opportunity.
