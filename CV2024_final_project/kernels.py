#%%

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
from util import *
#%%

np.resize

KERNEL_SIZE = 256

def generate_meshgrid(w,h):
    """
    w - width
    h - height
    """
    print(w,h)
    x = (np.linspace(-1, 1,w) * ((w - 1) / 2)).astype(np.float64)
    y = (np.linspace(-1, 1,h) * ((h - 1) / 2)).astype(np.float64)

    X,Y = np.meshgrid(x,y)

    return X,Y

def generate_distance_meshgrid(w,h):
    X,Y = generate_meshgrid(w,h)
    dist = np.sqrt(X**2 + Y**2)
    return dist

def  circle_parametric_kernel(radius, size = KERNEL_SIZE):
    """
    radius of the circle
    """
    dist = generate_distance_meshgrid(size, size)

    kernel = dist < radius
    kernel = kernel * 1.0
    kernel /= np.sum(kernel)

    return kernel


def line_parametric_kernel(length, angle = 0, size = KERNEL_SIZE):
    midlength = int(length / 2)
    midpoint = int(KERNEL_SIZE / 2)

    k = np.zeros((size, size))
    X,Y = generate_meshgrid(size, size)

    mask = np.abs(Y) < 1 & (np.abs(X) < midlength)
    
    k[mask] = 1
    k /= np.sum(k)

    k = transform_image(
        k,
        get_rotation_matrix_around_point(angle, midpoint, midpoint)
    )

    return k

def combine_kernels(k1, k2):
   shape = (max(k1.shape[0], k2.shape[0]), max(k1.shape[1], k2.shape[1]))
   k1 = np.fft.fftshift(k1)
   k1_pad = np.pad(k1, ((0, shape[0]-k1.shape[0]), (0, shape[1]-k1.shape[1])))
   k2_pad = np.pad(k2, ((0, shape[0]-k2.shape[0]), (0, shape[1]-k2.shape[1])))
   K1 = np.fft.fft2(k1_pad)
   K2 = np.fft.fft2(k2_pad) 
   K = K1*K2
   return np.fft.ifft2(K)

def gaussian_high_pass(x,D_0):
    return np.exp(-0.5 * ((x**2)) * (1 / (D_0 ** 2)))

def gaussian_kernel(sigma, size = KERNEL_SIZE):
    X,Y = generate_meshgrid(size, size)
    dist = np.sqrt(X**2 + Y**2)

    kernel = gaussian_high_pass(dist, sigma)

    kernel /= np.sum(kernel)

    return kernel

def make_image(k):
    k = np.array(k)
    k *= 255
    k = k.astype(np.uint8)
    return k

def make_kernel_image(k):
    k /= np.max(k)
    return make_image(k)


# show_original_with_ff(line_parametric_kernel(20,0))