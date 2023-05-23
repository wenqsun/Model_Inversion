echo "run black-box attack for FedMD and FedED on CIFAR100"

CUDA_VISIBLE_DEVICES=4 nohup python ideal.py --epochs=800 --save_dir=run/cifar100_3  --dataset=cifar100 --net=FL --g_steps=5 --exp_name=FedMD --batch_size=300 --acc 32.27 &

CUDA_VISIBLE_DEVICES=6 nohup python ideal.py --epochs=800 --save_dir=run/cifar100_4  --dataset=cifar100 --net=FL --g_steps=5 --exp_name=FedED --batch_size=300 --acc 27.54 --FL_algorithm=FedED
