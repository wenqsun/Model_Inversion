echo "white-box model inversion for FedAvg, FedProx, FedPAQ on CIFAR100"

echo "FedAvg"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 15000
wait

echo "FedProx"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 15000
wait

echo "FedPAQ"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 15000
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 15000
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 15000
wait

echo "use dense generator to conduct model inversion"
echo "FedAvg"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 15000 --data_num 5000 --use_dense
wait

echo "FedProx"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 15000 --data_num 5000 --use_dense
wait

echo "FedPAQ"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 15000 --data_num 5000 --use_dense
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 15000 --data_num 5000 --use_dense
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 15000 --data_num 5000 --use_dense
wait

echo "conduct model inversion for more epochs"
echo "FedAvg"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 20000
wait

echo "FedProx"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 20000
wait

echo "FedPAQ"
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 20000
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 20000
wait
CUDA_VISIBLE_DEVICES=4 nohup python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 20000


# CUDA_VISIBLE_DEVICES=4 python train_inversion.py --FL_algorithm FedPAQ --acc 60.85 --num_epoch 15000 --data_num 5000 --use_dense &

# wait
# # CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 10000
# wait
# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedProx --acc 67.87 --num_epoch 15000 --data_num 5000 --use_dense &
# # # acc=66.55?
# wait
# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 53.89 --num_epoch 15000 --data_num 5000 &
# wait
# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedPAQ --acc 65.18 --num_epoch 15000 --data_num 5000 &      # acc: 65.11

# CUDA_VISIBLE_DEVICES=6 python train_inversion.py --FL_algorithm FedAvg --acc 66.95 --num_epoch 15000 --use_dense &

