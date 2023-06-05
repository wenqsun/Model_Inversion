echo "run model inversion for FedAvg, FedProx, FedPAQ on SVHN"

# FedAVG
echo "dirichlet alpha = 0.01"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 88.47 --num_epoch 15000

wait

echo "dirichlet alpha = 0.1"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 90.55 --num_epoch 15000

wait

# FedProx
echo "dirichlet alpha = 0.01"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedProx --acc 89.23 --num_epoch 15000

wait

echo "dirichlet alpha = 0.1"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedProx --acc 90.23 --num_epoch 15000

wait

# FedPAQ
echo "dirichlet alpha = 0.01"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 85.64 --num_epoch 15000

wait

echo "dirichlet alpha = 0.1"
CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 88.34 --num_epoch 15000

wait



