# echo 'test the FID of FedMD and FedED'

# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedMD ||

# CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED ||

CUDA_VISIBLE_DEVICES=3 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 12000 &

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 10000

CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 12000 &
# # acc=66.55?

CUDA_VISIBLE_DEVICES=5 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 12000 &

CUDA_VISIBLE_DEVICES=6 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 12000  &      # acc: 65.11

CUDA_VISIBLE_DEVICES=7 nohup python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 12000 &

