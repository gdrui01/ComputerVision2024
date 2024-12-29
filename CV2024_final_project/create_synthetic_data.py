#%%
import glob
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
import scipy
import shutil

import os

import scipy.ndimage

from util import *
from kernels import *
#%%

IMG_SIZE = 512
blur_kernel = gaussian_kernel(0.7, IMG_SIZE)

def get_random_deformation_kernels():
    kernels = [
        line_parametric_kernel(random.randint(8,20) ,random.uniform(0, 180)),
        line_parametric_kernel(random.randint(8,20) ,random.uniform(0, 180)),
        line_parametric_kernel(random.randint(5,6) ,random.uniform(0, 180)),
        line_parametric_kernel(random.randint(30,32) ,45),
        line_parametric_kernel(random.randint(30,32) ,0),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),
        # line_parametric_kernel(random.randint(2,5) ,random.uniform(0, 180)),

        # circle_parametric_kernel(random.randint(5,20)),
        # circle_parametric_kernel(random.randint(5,20)),
        # circle_parametric_kernel(random.randint(5,20)),
        # circle_parametric_kernel(random.randint(5,20)),
        # circle_parametric_kernel(random.randint(5,20)),

        # combine_kernels(    
        #     line_parametric_kernel(random.randint(2,20) ,random.uniform(0, 360)),
        #     circle_parametric_kernel(random.randint(2,10)),
        # )
    ]
    
    # kernels = [np.fft.fftshift(k) for k in kernels]
    return kernels

def deform_image(path):
    name = Path(path).stem
    
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    img /= 255.0

    return kernel_deform_image(img, name)


def kernel_deform_image(img, name):
    
    # img = scipy.ndimage.convolve(img, blur_kernel, mode='constant', cval=0.0)



    # img = transform_image(img, get_scale_matrix(0.26, 0.26))
    img = degrade_image(img, np.fft.fftshift(blur_kernel), np.zeros((1,1)))
    

    kernels = get_random_deformation_kernels()    
    degraded_imgs = [scipy.ndimage.convolve(img, k, mode='reflect', cval=0.0) for k in kernels]
    # degraded_imgs = [img for k in kernels]


    for image_index, degraded_img in enumerate(degraded_imgs):    
        file_name = f"{name}_{image_index}"
    
        cv2.imwrite(f"images/synthetic_original/{file_name}.png", make_image(img))
        cv2.imwrite(f"images/synthetic_deformed/{file_name}.png", make_image(degraded_img))
        np.save(f"images/synthetic_kernel/{file_name}.npy", kernels[image_index])

        cv2.imwrite(f"images/synthetic_kernel_image/{name}_{image_index}_kernel.png",make_kernel_image(kernels[image_index]))

    return img, degraded_imgs, kernels

if __name__ == '__main__':

    
    shutil.rmtree("images/synthetic_deformed", ignore_errors=True)
    shutil.rmtree("images/synthetic_kernel", ignore_errors=True)
    shutil.rmtree("images/synthetic_original", ignore_errors=True)
    shutil.rmtree("images/synthetic_kernel_image", ignore_errors=True)

    os.makedirs("images/synthetic_deformed")
    os.makedirs("images/synthetic_kernel")
    os.makedirs("images/synthetic_original")
    os.makedirs("images/synthetic_kernel_image")


    image_paths = glob.glob("images/source/*.jpeg")

    plt.gray()

    for image_path in image_paths:

        print("deforming", image_path)

        
        img, degraded_images, kernels = deform_image(image_path)
        # plot_images([img,degraded_images[0], kernels[0]])
        

        # break
