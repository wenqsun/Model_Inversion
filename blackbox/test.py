import torch
from Model import MNIST_Net, InverseMNISTNet, InverseCIFAR10Net, FedAvgCNN, resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FedAvgCNN().to(device)
model_weight = torch.load('result_model/FedMD/dir_0.1_model_test.pth', map_location=device)

print(model)