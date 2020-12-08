import pixellib
from pixellib.semantic import semantic_segmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt

bg = int(input('Enter the background number you want?'))
background = cv2.imread(f"background{bg}.jpg")
segment_video = semantic_segmentation()
segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)
while True:
    retval, frame = cam.read()
    if retval != True:
        print("Can't read frame")
    else:
        _,frame1=segment_video.segmentAsPascalvoc(frame,process_frame=True)
        # frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seg_image = cv2.inRange(frame1,np.array([128,128,192]),np.array([128,128,192]))
        seg_image = cv2.cvtColor(seg_image,cv2.COLOR_BGR2RGB)
        seg_image = 255- seg_image
        # seg_image = frame1[:,:,0] > 100 
        # seg_image = np.uint8(seg_image)*255
        # seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
        # seg_image = 255 - seg_image
        back = np.copy(background)
        back = cv2.resize(back,(width,height))
        
        masked_image = cv2.resize(seg_image, (width,height))
        masked_image[seg_image != 0] = 0

        back[seg_image == 0] = 0
        frame[seg_image!=0]=0
        # print('comng to')
        full_image = back+frame
        cv2.imshow('frame',full_image)
        # plt.imshow(back+frame)
        # break
        if(cv2.waitKey(1)==27):
            break
cv2.destroyAllWindows()
cam.release()

