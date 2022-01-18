# Fairness-regularizers-in-semi-supervised-learning

Authors: **Tao Zhang**, Tianqing Zhu, Kun Gao, Wanlei Zhou

This is part of code for my paper 'Fairness in Graph-based Semi-supervised Learning'. 

Paper Link: https://arxiv.org/abs/2009.06190


## Introduction
Fairness in machine learning has received considerable attention. However, most studies on fair learning focus on either supervised learning or unsupervised learning. Very few consider semi-supervised settings. Yet, in reality, most machine learning tasks rely on large datasets that contain both labeled and unlabeled data.
Recent  study  has  proved  that  increasing  the  size  oftraining  (labeled)  data  will  promote  the  fairness  criteria  withmodel  performance  being  maintained.  In  this  work,  we  aim  toexplore  a  more  general  case  where  quantities  of  unlabeled  data are provided, indeed leading to a new form of learning paradigm,namely  fair  semi-supervised  learning.  Taking  the  popularity  ofgraph-based  approaches  in  semi-supervised  learning,  we  study this  problem  both  on  conventional  label  propagation  methodand  graph  neural  networks,  where  various  fairness  criteria  canbe  flexibly  integrated.  Our  developed  algorithms  are  provedto  be  non-trivial  extensions  to  the  existing  supervised  modelswith  fairness  constraints.  More  importantly,  we  theoretically demonstrate the source of discrimination in fair semi-supervised learning  problems  via  bias,  variance  and  noise  decomposition. Extensive  experiments  on  real-world  datasets  exhibit  that  ourmethods achieve a better trade-off between classification accuracy and  fairness  than  the  compared  baselines.

## Contribution
First, we conduct the study of algorithmic fairness in thesetting of graph-based SSL, including graph-based regularizations and graph neural networks. These approaches enable the use of unlabeled data to achieve a better trade-off between fairness and accuracy.

Second, we  propose  algorithms  to  solve  optimizationproblems  when  disparate  impact  and  disparate  mistreat-ment are integrated as fairness metrics in the graph-based regularization.

Third, we consider different cases of fairness constraintson labeled and unlabeled data. This helps us understandthe impact of unlabeled data on model fairness, and howto control the fairness level in practice.

Forth, we theoretically analyze the sources of discrimina-tion in SSL to explain why unlabeled data help to reacha  better  trade-off.  We  conduct  extensive  experiments  tovalidate the effectiveness of our proposed methods.


## Method 1


## Method 2
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
