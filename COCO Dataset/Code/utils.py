from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tqdm import tqdm
import image_slicer
import random
import matplotlib.pyplot as plt

def save_imgs(src_path,dest_path,className,coco,width,height):
    '''
    Developed by: Muhammad Saad (c)
    
    The function basically saves all the specific images to a folder. It filters out the images on the basis of the 
    class provided.
    
    Parameters:
        src_path (str): The path where the images are present. The source folder is input.
        dest_path (str): The path where you want to save the preprocessed images.
        className (str): The supercategory that will be applied to filter out the images.
        coco (obj): The module to conduct all the actions for the loading the images.
        img_input_size (tuple): The desired image size. It has to be width x height only
    
    Returns:
        unique_images (dict): The dictionary of all the images that are unique of the specified class
        catIds (list) : The list of the ids that are extracted from the class specified 
        
        The images are also saved into the specified folder
    '''
    path_old = os.getcwd()
    os.chdir(src_path)
    classes = [className]

    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in tqdm(classes):
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images    
    unique_images = []
    for i in tqdm(range(len(images)),desc='Image Segregation'):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            

    dataset_size = len(unique_images)

    print("Number of images containing the filter classes:", dataset_size)
    
    ## Saving the images to the destination folder
    
    for i in tqdm(range(len(unique_images)),desc='Saving Images'):
        img=cv2.resize(cv2.imread(unique_images[i]['file_name']),(width,height))
        cv2.imwrite(os.path.join(dest_path,unique_images[i]['file_name']),img)
    
    os.chdir(path_old)
    
    return unique_images,catIds

def save_ann(dest_path,unique_images,coco,cat_ids,width,height):
    '''
    Developed by: Muhammad Saad (c)
    
    The module saves the annotations to any specified folder. 
    
    Parameters:
        dest_path (str): The path where you want to save the annotations
        unique_images (dict): The dictionary of all the unique images of a specific class. 
        coco (obj) : The object for the coco which has all the variables
        cat_ids (list): The list of ids that are being extracted from the class
    
    Return:
        The images are saved into the specified folder
    '''
    path_old = os.getcwd()
    os.chdir(dest_path)
    print(f'The total images are {len(unique_images)}')
    for i in tqdm(range(len(unique_images)),desc='Annotation Saving'):
        img=unique_images[i]['id'] # Extracting the id of the image
        filename =unique_images[i]['file_name']
        ann_id=coco.getAnnIds(img, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_id)
        mask = np.zeros((height,width))
        if len(anns)>1:
            continue
        else:
            new_mask = cv2.resize(coco.annToMask(anns[0]), (width,height))
            new_mask[new_mask >= 0.5] = 1
            new_mask[new_mask < 0.5] = 0
            mask = np.maximum(new_mask, mask)
            cv2.imwrite(filename,cv2.convertScaleAbs(mask, alpha=(255.0)))
    os.chdir(path_old)
    return

def cleanup(target_path,reference_path):
    count=0
    for i in os.listdir(target_path):
        if i not in os.listdir(reference_path):
            os.remove(os.path.join(target_path,i))
            count+=1
    print(f'{count} files removed')
    return

def slicer(path,slices):
    path_old = os.getcwd()
    os.chdir(path)
    for i in sorted(os.listdir()):
        temp = image_slicer.slice(i,slices)
        os.remove(i)
    
    os.chdir(path_old)
    
    return

def read_imgs(path):
    img,names = [],[]
    for i in sorted(os.listdir(path)):
        names.append(i)
        imgs=cv2.imread(os.path.join(path,i))
        img.append(cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
    return names,np.array(img)

def visualize(in_images,out_images,number):
    for i in number:
        plt.subplot(121)
        plt.imshow(in_images[i])
        plt.title('Original')
        
        plt.subplot(122)
        plt.imshow(out_images[i])
        plt.title('Output')
        
        plt.show()
    
def make_blur(img):
    # image = cv2.imread(img)
    ksize = (10,10)
    blur = cv2.blur(img, ksize)
    return blur

def mask_blur(original_img, blur_img, predicted_img,width,height):
    # print("Shape: ", original_img.shape, blur_img.shape, predicted_img.shape)
    # cv2.imshow('img', predicted_img)
    # cv2.waitKeyEx(0)

    blue_channel_ori = original_img[:, :, 0]
    green_channel_ori = original_img[:, :, 1]
    red_channel_ori = original_img[:, :, 2]

    blue_channel_blr = blur_img[:, :, 0]
    green_channel_blr = blur_img[:, :, 1]
    red_channel_blr = blur_img[:, :, 2]

    blue_channel_pre = predicted_img[:, :, 0]
    green_channel_pre = predicted_img[:, :, 1]
    red_channel_pre = predicted_img[:, :, 2]

    new_b = []
    new_g = []
    new_r = []

    mks_img_new = np.zeros([width,height, 3])

    for i in range(3):
        if i == 0:
            img = blue_channel_blr
            msk = blue_channel_pre
            ori = blue_channel_ori
        if i == 1:
            img = green_channel_blr
            msk = green_channel_pre
            ori = green_channel_ori
        if i == 2:
            img = red_channel_blr
            msk = red_channel_pre
            ori = red_channel_ori

        if i == 0:
            new = new_b
        if i == 1:
            new = new_g
        if i == 2:
            new = new_r

        img = img.reshape(1, -1)[0]
        msk = msk.reshape(1, -1)[0]
        ori = ori.reshape(1, -1)[0]

        for k, m, o in zip(img, msk, ori):
            if int(m*255.) < 50:
                new.append(k)
            else:
                new.append(o)

        if i == 0:
            new_b = np.array(new_b).reshape(width, height)
            mks_img_new[:, :, 0] = new_b
        if i == 1:
            new_g = np.array(new_g).reshape(width, height)
            mks_img_new[:, :, 1] = new_g
        if i == 2:
            new_r = np.array(new_r).reshape(width, height)
            mks_img_new[:, :, 2] = new_r

    return mks_img_new


def create_img(original_img, predicted_img,width,height):
    blur_img = make_blur(original_img)
    img = mask_blur(original_img, blur_img, predicted_img,width,height)
    
    return img