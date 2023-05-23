# echo 'test the FID of FedMD and FedED'

# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedMD ||

# CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED ||

CUDA_VISIBLE_DEVICES=5 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 &

CUDA_VISIBLE_DEVICES=7 nohup python train_inversion.py --FL_algorithm FedAvg --acc 66.95