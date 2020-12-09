from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import utils
from torchvision import models


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

bg = int(input('Enter the background number you want?'))
background = cv2.imread(f"background{bg}.jpg")

cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)
while True:
    retval, frame = cam.read()
    if retval != True:
        print("Can't read frame")
    else:
        rgb = utils.segment(dlab,None,frame,False)
        rgb[rgb!=255]=0
        height,width,channel = rgb.shape
        
        alpha = rgb
        alpha = alpha.astype(float)/255
        
        back = np.copy(background)
        back = cv2.resize(back,(width,height),interpolation = cv2.INTER_AREA)

        fore = frame.astype(float)
        back = back.astype(float)

        fore = cv2.multiply(alpha, fore)
        back = cv2.multiply(1.0 - alpha, back)
        outImage = cv2.add(fore, back)

        cv2.imshow('frame',outImage)
        if(cv2.waitKey(1)==27):
          break
cv2.destroyAllWindows()
cam.release()



