from __future__ import print_function, division
import torch
from torchvision.transforms import transforms
from PIL import Image
import sys
import cv2
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

name = sys.argv[1]

#from pathlib import Path

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)
try:
	model_ft.load_state_dict(torch.load('./snapshot.pth'))
	
except:
	print("couldnt load model")
model_ft.eval()   # Set model to evaluate mode

trans = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
try:
	image = Image.open(name)
except:
	print('couldnt open file')

input = trans(image)

input = input.view(1, 3, 224,224)

output = model_ft(input)
print(output)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

if (prediction == 0):
	print ('glass')
if (prediction == 1):
	print ('leather')
if (prediction == 2):
	print ('metal')
if (prediction == 3):
	print ('wood')