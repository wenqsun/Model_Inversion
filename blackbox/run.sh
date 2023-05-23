echo "run blackbox attack for FedMD and FedED on CIFAR10"

# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=800 --save_dir=run/cifar10_3  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_local_soft --batch_size=250 --acc 47.78

# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=400 --save_dir=run/cifar10_3  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 47.78 &

# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=400 --save_dir=run/cifar10_4  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_local_soft_CE --batch_size=250 --acc 47.78 --local &

# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=400 --save_dir=run/cifar10_5  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 43.44 &

# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=400 --save_dir=run/cifar10_6  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 48.29 &

# CUDA_VISIBLE_DEVICES=0 python ideal.py --epochs=400 --save_dir=run/cifar10_7  --dataset=cifar10 --net=FL --g_steps=5 --FL_algorithm FedED --exp_name=FedED_global_soft_CE --batch_size=250 --acc 44.17 &

# CUDA_VISIBLE_DEVICES=0 python ideal.py --epochs=400 --save_dir=run/cifar10_8  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 43.55 --FL_algorithm FedMDDP --dp_level 0.01&

# CUDA_VISIBLE_DEVICES=0 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_3  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 43.55 --FL_algorithm FedMDDP --dp_level 0.1&

# CUDA_VISIBLE_DEVICES=0 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_3  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD_global_soft_CE --batch_size=250 --acc 43.55 --FL_algorithm FedMDDP --dp_level 1.0&

# echo "FedMD with noise"
# CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=800 --save_dir=run/cifar10_1  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMD --batch_size=250 --acc 31.76 --FL_algorithm FedMD &

echo "FedMD with DP noise"
CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=800 --save_dir=run/cifar10_1  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedMDDP --batch_size=250 --acc 43.55 --FL_algorithm FedMDDP --dp_level 0.01 &

echo "FedED with no defense"
CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=800 --save_dir=run/cifar10_2  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedED --batch_size=250 --acc 43.35 --FL_algorithm FedED &

# echo "FedED with local client"
# CUDA_VISIBLE_DEVICES=2 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_3  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedED_local --batch_size=250 --acc 43.35 --FL_algorithm FedED --local &

# echo "FedED with DP defense"
# CUDA_VISIBLE_DEVICES=2 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_4  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedEDDP --batch_size=250 --acc 35.93 --FL_algorithm FedEDDP --dp_level 0.01 &

# echo "FedED with noise defense"
# CUDA_VISIBLE_DEVICES=2 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_5  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedED --batch_size=250 --acc 36.9 --FL_algorithm FedED &

# echo "FedED with only data augmentation defense"
# CUDA_VISIBLE_DEVICES=2 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_6  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedED --batch_size=250 --acc 43.79 --FL_algorithm FedED &

# echo "FedMD with local mixup"
# CUDA_VISIBLE_DEVICES=2 nohup python ideal.py --epochs=800 --save_dir=run/cifar10_7  --dataset=cifar10 --net=FL --g_steps=5 --exp_name=FedED --batch_size=250 --acc 39.5 --FL_algorithm FedED &

