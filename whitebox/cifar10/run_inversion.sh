echo "White Model Inversion attack for FedPAQ and FedPAQDP on CIFAR10"

CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --local --acc 58.82 --num_epoch 10000

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQDP --dp_level 0.01

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQ --acc 58.82 &

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQDP --acc 56.66 --dp_level 0.01 &

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQDP --acc 56.66 --dp_level 0.01    # try different seeds

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQ --local --acc 58.82

# echo "data agumentation for FedPAQ"

# CUDA_VISIBLE_DEVICES=5 python train_inversion.py --FL_algorithm FedPAQ --acc 61.45 &

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 61.74 &

# CUDA_VISIBLE_DEVICES=7 python train_inversion.py --FL_algorithm FedPAQ --acc 47.31 
