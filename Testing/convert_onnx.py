from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import utils
from torchvision import models
from pytorch2keras.converter import pytorch_to_keras

x = torch.randn(1,3,224, 224, requires_grad=False)

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

# pytorch_to_keras(dlab, x, [(3,None, None,)], verbose=True, name_policy='short')

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(dlab,x,'segmentation.onnx',verbose=True,input_names=input_names,
                  output_names=output_names)