# echo 'test the FID of FedMD and FedED'

# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedMD ||

# CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED ||

CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 15000 --data_num 5000 --use_dense &

wait
# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 10000
wait
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 15000 --data_num 5000 --use_dense &
# # acc=66.55?
wait
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 15000 --data_num 5000 --use_dense &
wait
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 15000 --data_num 5000 --use_dense &      # acc: 65.11

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 15000 --use_dense &

