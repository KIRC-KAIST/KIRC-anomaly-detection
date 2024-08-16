This folder contains the data and source code required to train the model proposed by HyunSoo Kim in the thesis "Aggressive Behavior Detection in pigs based on Reconstruction Loss Inversion and adversarial attack"
The code folder contains the model to be run for training.
The data folder contains the data needed for training
For training, you should have an Ubuntu 20.04 or later system with CUDA support and Python 3.8.

Rum the following commands for training
1- cd code/
2- pip install -r requirements.txt
3- !python train.py --exp Inv_FGSM_ep_0.1 --eps 0.1 --FGSM True --fgsm_type feature --train exp_4_train --test exp_4_test --gpu 0
