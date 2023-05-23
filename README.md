# Model Inversion Attack
This is a code implementation for model inversion attack, both in white-box and black-box settings.

## Update Log
### 21 Apr
Black-box attack implementation. reference: https://github.com/yziqi/adversarial-model-inversion/tree/master

### 23 Apr
Complete the black-box attack implementation. Please find the folder blackbox, and run `python train_inversion.py --save-model --epochs 50 --lr 0.0005 --log-interval 100`.

Complete the white-box attack implementation. Please find the folder whitebox, and run `python train_inversion.py --num_epoch 200`

Up to now, we assume that we have a generic public dataset to train our GAN and inversion model.


### 2 May
TODO: Update the model inversion attack on CIFAR10, CIFAR100 datasets

Due to an illness, I will complete this task later.

### 8 May
Complete white-box model inversion attack for CIFAR10 and CIAFAR100 datasets.

### 23 May
In the `\blackbox` folder, you can find the main code `ideal.py` for FedMD and FedED on cifar10 and cifar100 datasets. Just refer the `run.sh` for the usage. Also, you need the targeted model for different methods, which are placed in the `\pretrained` folder. The model required can be downloaded here: 

In the `\whitebox` folder, you can find `cifar10` and `cifar100` folders. Each folder contains code for 
