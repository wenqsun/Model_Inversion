echo "FID for FL algorithms"

echo "-------------- FedPAQ --------------"

CUDA_VISIBLE_DEVICES=6 python metric.py --method FedPAQ --acc 53.89

echo "-------------- FedAvg --------------"

CUDA_VISIBLE_DEVICES=6 python metric.py --method FedAvg --acc 66.85