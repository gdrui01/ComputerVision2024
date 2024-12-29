#%%
import numpy as np
from scipy import signal, ndimage
from PIL import Image

def pad_image(f, kernel_size):
    """Pad input image to handle convolution boundary effects"""
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), 'reflect')

def normalize_kernel(k):
    """Normalize kernel to sum to 1"""
    return k / np.sum(k)

def blind_deconvolution(f, kernel_size, lambda_start=1, lambda_min=0.0006, max_iters=100):
    """
    Blind deconvolution algorithm based on the paper.
    
    Parameters:
    -----------
    f : 2D numpy array
        Blurry input image
    kernel_size : tuple
        Size of the blur kernel (h, w)
    lambda_start : float
        Initial regularization parameter
    lambda_min : float
        Minimum regularization parameter
    max_iters : int
        Maximum number of iterations
    """
    # Initialize
    u = pad_image(f, kernel_size)  # Initialize u0 with padded input
    k = np.ones(kernel_size) / np.prod(kernel_size)  # Initialize k0 as uniform
    lambda_val = lambda_start
    
    # Gradient descent parameters
    epsilon_u = 0.01  # Step size for u update
    epsilon_k = 0.01  # Step size for k update
    
    for t in range(max_iters):
        # Update u using gradient descent
        conv_uk = signal.convolve2d(u, k, 'same')
        diff = conv_uk - f
        grad_u = signal.convolve2d(diff, np.rot90(k, 2), 'same')
        
        # Calculate TV regularization term
        u_grad = np.gradient(u)
        u_grad_norm = np.sqrt(u_grad[0]**2 + u_grad[1]**2 + 1e-10)
        tv_term = -np.gradient(u_grad[0] / u_grad_norm)[0] - np.gradient(u_grad[1] / u_grad_norm)[1]
        
        # Update u
        u = u - epsilon_u * (grad_u - lambda_val * tv_term)
        
        # Update k using gradient descent
        kt = k
        conv_u_new = signal.correlate2d(u, diff, 'valid')
        k = k - epsilon_k * conv_u_new
        
        # Project k onto constraints
        k = np.maximum(k, 0)  # Non-negativity
        k = normalize_kernel(k)  # Sum-to-one constraint
        
        # Update lambda
        lambda_val = max(0.99 * lambda_val, lambda_min)
        
        # Check convergence
        if np.all(np.abs(k - kt) < 1e-6):
            break
            
    return u, k

img = np.array(Image.open('images/synthetic_deformed/01_0.png').convert('L')) / 255.0# your blurry image
kernel_size = (15, 15)  # Size of blur kernel to estimate
deblurred_img, estimated_kernel = blind_deconvolution(img, kernel_size)