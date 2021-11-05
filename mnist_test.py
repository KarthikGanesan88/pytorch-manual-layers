import pdb

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor

from approximate_convolution_layer import MyConv2d
from approximate_fully_connected_layer import MyLinear

import numpy as np

np.set_printoptions(linewidth=10000)  # To not wrap print
np.set_printoptions(threshold=np.inf)  # To not truncate arrays

# To ignore the warning about not using max_pool2d.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 0: Define the neural network model, return logits instead of activation in forward method
class exact_model(nn.Module):
    def __init__(self):
        super(exact_model, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class appx_model(nn.Module):
    def __init__(self):
        super(appx_model, self).__init__()
        self.conv_1 = MyConv2d(1, 4, 5, stride=1, padding=0)
        self.conv_2 = MyConv2d(4, 10, 5, stride=1, padding=0)
        self.fc_1 = MyLinear(4 * 4 * 10, 100)
        self.fc_2 = MyLinear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 10) view throws an error so had to use re-shape.
        x = x.reshape(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
batch_size = 4

trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=False, transform=transform)

# Use test subset to change the number of inputs for inference runs.
test_subset = torch.utils.data.Subset(testset, range(0, 10000))
testloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

if torch.cuda.is_available():
    exact_net = exact_model().cuda()
else:
    exact_net = exact_model()

PATH = './data/pytorch_mnist.pth'
exact_net.load_state_dict(torch.load(PATH, map_location=device))

print(f"model_exact is on: {next(exact_net.parameters()).device}")

if torch.cuda.is_available():
    appx_net = appx_model().cuda()
else:
    appx_net = appx_model()

appx_net.load_state_dict(exact_net.state_dict())
print(f"appx_net is on: {next(appx_net.parameters()).device}")

if torch.cuda.is_available():
    quant_net = exact_model().cuda()
else:
    quant_net = exact_model()

# Code to get a quantized net using the pytorch-quantization-toolkit from Nvidia.
# This only does 'fake quantization' where the weights stay as float but based
# on the max value per layer, each fp32 weight is set to the nearest int8 equivalent.

# quant_desc_input = QuantDescriptor(calib_method='histogram')
# quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
#
# quant_modules.initialize()
# with torch.no_grad():
#     for layer in exact_net.state_dict():
#         # print(layer)
#         absolute = exact_net.state_dict()[layer][:].abs().max()
#         quant_net.state_dict()[layer][:] = tensor_quant.fake_tensor_quant(exact_net.state_dict()[layer][:], absolute)
#
# PATH = './data/pytorch_mnist_quant.pth'
# torch.save(quant_net.state_dict(), PATH)

quant_net.load_state_dict(torch.load('./data/pytorch_mnist_quant.pth', map_location=device))

model_list = ['exact_net', 'appx_net', 'quant_net']
correct = [0, 0, 0]
total = [0, 0, 0]

with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Exact model
        outputs = exact_net(images)
        _, predicted_exact = torch.max(outputs.data, 1)
        total[0] += labels.size(0)
        correct[0] += (predicted_exact == labels).sum().item()

        # Appx model
        outputs = appx_net(images)
        _, predicted_exact = torch.max(outputs.data, 1)
        total[1] += labels.size(0)
        correct[1] += (predicted_exact == labels).sum().item()

        # Quant model
        outputs = quant_net(images)
        _, predicted_exact = torch.max(outputs.data, 1)
        total[2] += labels.size(0)
        correct[2] += (predicted_exact == labels).sum().item()

    for i in range(3):
        print('Accuracy of ' + model_list[i] + ': %f %%' % (100 * correct[i] / total[i]))



