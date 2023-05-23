echo "compute FID score for FePAQ"

# CUDA_VISIBLE_DEVICES=7 python metric.py --method FedAvg --acc 66.81

# wait

# CUDA_VISIBLE_DEVICES=7 python metric.py --method FedAvg --acc 68.8

# wait

# CUDA_VISIBLE_DEVICES=7 python metric.py --method FedAvg --acc 69.92

# wait

# CUDA_VISIBLE_DEVICES=7 python metric.py --method FedProx --acc 68.53

# echo '||global model on local test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 58.82 --local_test

# wait

# echo '||local model on local test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 58.82 --local_test --local

# wait

# echo '||global model on global test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 58.82

# wait

# echo '||FedPAQDP on global test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQDP --acc 58.82 --dp_level 0.01

# wait

# echo '||FedPAQ with noise on global test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 47.31

# wait

# echo '||FedPAQ with only data augmentation on global test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 61.45

# wait

# echo '||FedPAQ with only local mixup on global test set||'
# CUDA_VISIBLE_DEVICES=5 python metric.py --method FedPAQ --acc 61.74

# wait

echo "------------------------- FID for FedMD -------------------------"

echo '||global model on local test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 47.51 --local_test

wait

echo '||local model on local test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 47.51 --local_test --local

wait

echo '||global model on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 47.51

wait

echo '||FedMD with DP noise on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMDDP --acc 47.51 --dp_level 0.01

wait

echo '||FedMD with noise on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 31.76

wait

echo '||FedMD with only data augmentation on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 48.29

wait

echo '||FedMD with only local mixup on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedMD --acc 43.44


echo "------------------------- FID for FedED -------------------------"

echo '||global model on local test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 43.35 --local_test 

wait

echo '||local model on local test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 43.35 --local_test --local

wait

echo '||global model on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 43.35

wait

echo '||FedED with DP noise on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedEDDP --acc 43.35 --dp_level 0.01

wait

echo '||FedED with noise on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 36.9

wait

echo '||FedED with only data augmentation on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 43.79

wait

echo '||FedED with only local mixup on global test set||'
CUDA_VISIBLE_DEVICES=6 python metric.py --method FedED --acc 39.5
