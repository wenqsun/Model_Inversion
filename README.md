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
In the `\blackbox` folder, you can find the main code `ideal.py` for FedMD and FedED on cifar10 and cifar100 datasets. Just refer the `run.sh` for the usage. And `metric.py` is used to compute the FID score. Also, you need the targeted model for different methods, which are placed in the `\pretrained` folder. The model required can be downloaded here: https://drive.google.com/drive/folders/1Cqkz4aGR5orTFgbJeVHgEOyS4ToEXe2-?usp=share_link

In the `\whitebox` folder, you can find `cifar10` and `cifar100` folders. Each folder contains code for different methods. Just refer the `run.sh` for the usage. And `metric.py` is used to compute the FID score. Also, you need the targeted model for different methods, which are placed in the `\result_model` folder. The model required can be downloaded here: 

cifar10: https://drive.google.com/drive/folders/1ZqEcc4I5vVTzvSN2xi1IXrF5vTNZht3z?usp=share_link

cifar100: https://drive.google.com/drive/folders/1zPqEt4pp7fL4yGrFnmAGzIlSsaDYZJ2L?usp=sharing

whitebox attack is highly referenced from https://github.com/MKariya1998/GMI-Attack

blackbox attack is high referenced from https://github.com/SonyResearch/IDEAL


### 24 May
Fix some errors in the code `\whitebox\cifar100\train_inversion.py`, mainly about the random noise inputs and generator. Now you can find the new code in `\whitebox\cifar100\train_inversion.py`. 

Now you can refer to `\whitebox\cifar100\train_inversion.py` to directly obtain the FID score.