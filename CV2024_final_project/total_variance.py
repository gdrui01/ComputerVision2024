#%%

import numpy as np
from scipy import signal
import time
from PIL import Image
from matplotlib import pyplot as plt
from util import *
import util

#%%

def dTV(x):
    dx = np.pad(np.diff(x, axis=1), ((0, 0), (0, 1)), mode='reflect')
    dy = np.pad(np.diff(x, axis=0), ((0, 1), (0, 0)), mode='reflect')
    
    epsilon = 1e-10
    norm = np.sqrt(dx**2 + dy**2 + epsilon)
    
    dx_norm = dx / norm
    dy_norm = dy / norm
    
    div_x = np.pad(-np.diff(dx_norm, axis=1), ((0, 0), (1, 0)), mode='reflect')
    div_y = np.pad(-np.diff(dy_norm, axis=0), ((1, 0), (0, 0)), mode='reflect')
    
    return div_x + div_y

def TVdeblur(image, alpha1, alpha2, gamma, beta, iterations):
    un = image.copy()
    kn = np.zeros((13, 13))
    
    kn_size = kn.shape
    pad_h = (kn_size[0] - 1) // 2
    pad_w = (kn_size[1] - 1) // 2
    z = image[pad_h:-pad_h, pad_w:-pad_w]
    
    time_sum = 0
    
    for n in range(iterations):
        print(np.all(np.isnan(un)))
        start_time = time.time()
        
        conv_term = signal.convolve2d(un, kn, mode='valid') - z
        dkn = signal.convolve2d(np.rot90(np.rot90(un)), conv_term, mode='valid') - alpha2 * dTV(kn)
        
        kn = kn - beta * dkn

        
        
        kn = np.maximum(kn, 0)  
        kn = (kn + np.rot90(np.rot90(kn))) / 2
        kn = kn / np.sum(kn)
        
        conv_term = signal.convolve2d(un, kn, mode='valid') - z
        dun = signal.convolve2d(conv_term, np.rot90(np.rot90(kn))) - alpha1 * dTV(un)
        
        # un = un - gamma * dun
        # un = np.maximum(un, 0)
        
        elapsed_time = time.time() - start_time
        time_sum += elapsed_time
        time_avg = time_sum / (n + 1)
        
        if (n + 1) % 10 == 0:
            percent_complete = (n + 1) / iterations * 100
            time_remaining = (iterations - (n + 1)) * time_avg / 60
            print(f"Percent Complete: {percent_complete:.2f}%")
            print(f"Estimated Time Remaining: {time_remaining:.2f} minutes")
            
    return un, kn



og_image = np.array(Image.open('images/synthetic_original/01_0.png').convert('L')) / 255.0
image = np.array(Image.open('images/synthetic_deformed/01_0.png').convert('L')) / 255.0
og_kernel = np.load("images/synthetic_kernel/01_0.npy")

alpha1 = 1e-3
alpha2 = 1e-3
gamma = 0.08
beta = 0.08
iterations = 20

deblurred_image, estimated_kernel = TVdeblur(image, alpha1, alpha2, gamma, beta, iterations)

plt.gray()
plot_images([
    og_image,
    image, 
    np.fft.ifftshift(util.wiener_deconvolution(image,estimated_kernel))
    ])

plot_images([
    estimated_kernel,
    og_kernel,
    ])

# %%
