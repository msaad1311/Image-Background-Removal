import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T

def decode_segmap(image, nc=21):

    # 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, # 11=dining table, 12=dog
    # 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    label_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (255, 255, 255), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (255, 255, 255),(0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def segment(net, path=None, frame=None, show_orig=False, dev='cuda'):
    if path!=None:
        img = Image.open(path)
    else:
        img = frame
    if show_orig: 
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        

    if isinstance(img,np.ndarray):
        img = Image.fromarray(img)
    trf = T.Compose([T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])#, T.Resize(640), T.CenterCrop(224),
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    # plt.imshow(rgb); plt.axis('off'); plt.show()
    # cv2.imwrite('mask2.png' , rgb)
    #   files.download('mask2.png')
    return rgb

def bg_change(seg,frame,bg):
    if bg == 1: # cropped image
        outImage=cv2.subtract(seg,frame)
        outImage=cv2.subtract(seg,outImage)
        outImage[seg == 0] = 255
        outImage = np.array(outImage,dtype=float)/float(255)
        
        return outImage
    
    elif bg in [2,3]: # blurred image
        if bg==2:
            background = cv2.GaussianBlur(frame, (31,31), 0)
        elif bg==3:
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        
        foreground = frame.astype(float)
        background = background.astype(float)

        th, alpha = cv2.threshold(np.array(seg),0,255, cv2.THRESH_BINARY)
        alpha = cv2.GaussianBlur(alpha, (7,7),0)
        alpha = alpha.astype(float)/255
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        outImage = np.array(outImage,dtype=float)/float(255)
        
        return outImage
    
    else:
        alpha = seg
        height,width,channel = seg.shape
        alpha = alpha.astype(float)/255
        
        background = cv2.imread(f"background{bg}.jpg")
        background = cv2.resize(background,(width,height),interpolation = cv2.INTER_AREA)
        
        foreground = frame.astype(float)
        background = background.astype(float)

        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        outImage = np.array(outImage,dtype=float)/float(255)
        return outImage
        