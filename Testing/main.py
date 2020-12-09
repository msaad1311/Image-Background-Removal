from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import utils
from torchvision import models

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

bg = int(input('Enter the background number you want?'))


cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)
while True:
    retval, frame = cam.read()
    if retval != True:
        print("Can't read frame")
    else:
        rgb = utils.segment(dlab,None,frame,False)
        rgb[rgb!=255]=0
        output = utils.bg_change(rgb,frame,bg)
        cv2.imshow('frame',output)
        # break
        if(cv2.waitKey(1)==27):
          break
cv2.destroyAllWindows()
cam.release()



