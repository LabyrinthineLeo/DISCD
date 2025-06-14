### Rethinking Learner Modeling: A Feedback-Centric Cognitive Disentanglement Perspective

***

:clipboard: The code in this repository is the implementation of the proposed DISCD model.

#### Introduction

Cognitive diagnosis (CD) has been always recognized a essential learner modeling task in personalized education, which aims to infer learners’ mastery in specific knowledge concepts by mining and analyzing their practice behavior. However, most existing studies fail to explicitly disentangling the multiple interdependent factors that influence learner’s response feedback during the problem-solving process, both in web-based environments and real-world contexts. To address this issue, we propose DISCD, a feedback-centric DISentangled Cognitive Diagnosis framework for enhancing effective and interpretable learner modeling. 

#### Dependencies
* python 3.8
* pytorch 1.13+cu117
* numpy
* pandas
* sklearn


#### Usage
Train & Test model:
```
lrs=(1e-2 8e-3 5e-3 3e-3 1e-3)
python3 ./train.py --model "discd" --save_dir "../Results" --epoch 30 --dataset "nips-edu" --lr $lr --lambda_1 1e-5
```