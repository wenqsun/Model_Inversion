# echo "White Model Inversion attack for FedPAQ and FedPAQDP on CIFAR10"

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --local --acc 58.82 --num_epoch 10000

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQDP --dp_level 0.01

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQ --acc 58.82 &

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQDP --acc 56.66 --dp_level 0.01 &

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQDP --acc 56.66 --dp_level 0.01    # try different seeds

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQ --local --acc 58.82

# echo "data agumentation for FedPAQ"

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQ --acc 61.45 &

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 61.74 &

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQ --acc 47.31 

echo "FedAvg, FedProx, FedPAQ, FedAvgDP and FedPAQDP on CIFAR10"

CUDA_VISIBLE_DEVICES=0 python train_inversion.py --FL_algorithm FedAvg --acc 66.81 --num_epoch 15000 --log_interval 500 &

CUDA_VISIBLE_DEVICES=1 python train_inversion.py --FL_algorithm FedProx --acc 68.53 --num_epoch 15000 --log_interval 500 &

wait

CUDA_VISIBLE_DEVICES=2 python train_inversion.py --FL_algorithm FedPAQ --acc 61.72 --num_epoch 15000 --log_interval 500 &

CUDA_VISIBLE_DEVICES=3 python train_inversion.py --FL_algorithm FedPAQ --acc 47.31 --num_epoch 15000 --log_interval 500 &

wait

CUDA_VISIBLE_DEVICES=4 python train_inversion.py --FL_algorithm FedPAQDP --acc 57.74 --num_epoch 15000 --log_interval 500 &

CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedAvgDP --acc 61.45 --num_epoch 15000 --log_interval 500 &

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedAvg --acc 66.81 --num_epoch 5000 --log_interval 200 &
