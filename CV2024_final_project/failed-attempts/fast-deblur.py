#%%
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
import numpy as np
import cv2

class ImageLogger:
    def __init__(self):
        self.images = []
        self.labels = []
    
    def log(self, image, label):
        """Log an image with its label"""
        self.images.append(np.array(image))
        self.labels.append(label)
    
    def display(self, figsize=(15, 5)):
        """Display all logged images in a single figure"""
        n = len(self.images)
        if n == 0:
            print("No images logged")
            return
        
        # Calculate grid dimensions
        cols = min(4, n)  # Max 4 images per row
        rows = (n + cols - 1) // cols
        
        # Create figure
        plt.figure(figsize=(figsize[0], figsize[1] * rows/2))
        
        # Plot each image
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.title(label)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def clear(self):
        """Clear all logged images"""
        self.images = []
        self.labels = []

logger = ImageLogger()

def build_pyramid(image, num_scales):
    """Build Gaussian pyramid of the image"""
    pyramid = [image]
    # logger.log(image,f"pyramid {0}")
    current = image
    for i in range(num_scales - 1):
        current = cv2.resize(current, None, fx=0.5, fy=0.5)
        pyramid.append(current)
        # logger.log(current,f"pyramid {i+1}")
    # Return pyramid from coarsest to finest
    return pyramid[::-1]

def deblur_image(blurred_image, kernel_size=13, num_scales=4, iterations_per_scale=7):
    """Main deblurring process with pre-built pyramid"""
    
    # Build image pyramid from coarsest to finest
    pyramid = build_pyramid(blurred_image, num_scales)
    
    # Initialize kernel at coarsest scale
    current_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Process each scale
    for scale, current_image in enumerate(pyramid):
        print(f"Processing scale {scale}, image shape: {current_image.shape}")
        
        for iteration in range(iterations_per_scale):

            # Step 1: Prediction
            # edges = predict_sharp_edges(current_image)
            # logger.log(current_image,"current img")

            # logger.log(edges[0],"edg")
            # logger.log(edges[1],"edg2")
            
            # Step 2: Kernel Estimation using JAX
            current_kernel = estimate_kernel_jax(
                current_image,  # use image directly 
                current_image,  # instead of edge maps
                current_kernel.shape[0]
            )
            
            logger.log(current_image,"img")

            # Step 3: Deconvolution
            current_image = deconvolve(current_image, current_kernel)



            # return np.zeros((10,10)), np.zeros((10,10))
        
        # Prepare kernel for next scale (if not at finest scale)
        if scale < len(pyramid) - 1:
            current_kernel = cv2.resize(
                current_kernel, 
                (kernel_size, kernel_size),
                interpolation=cv2.INTER_LINEAR
            )
            current_kernel = current_kernel / current_kernel.sum()
    
    return current_image, current_kernel

def predict_sharp_edges(image):
    """Step 1: Edge Prediction"""
    # Bilateral filter for noise reduction
    smoothed = cv2.bilateralFilter(
        image.astype(np.float32), 
        d=5, 
        sigmaColor=0.5, 
        sigmaSpace=2.0
    )

    logger.log(image, 'og')
    logger.log(smoothed, 'bilateral')
    
    # Compute gradients
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    logger.log(grad_x, 'grad_x')
    logger.log(grad_y, 'grad_y')

    # Threshold gradients
    magnitude_x = np.sqrt(grad_x**2)
    magnitude_y = np.sqrt(grad_y**2)

    # logger.log(magnitude, 'magn')

    threshold_x = np.percentile(magnitude_x, 90)
    threshold_y = np.percentile(magnitude_y, 90)
    # threshold_x = np.percentile(magnitude, 90)
    mask_x = magnitude_x > threshold_x
    mask_y = magnitude_y > threshold_y

    
    grad_x[~mask_x] = 0
    grad_y[~mask_y] = 0
    
    return np.stack([grad_x, grad_y])

@jit
def energy_function(kernel, latent_image, blurred_image, beta=0.5):
    """Chan and Wong energy function: ||K * L - B||² + β||∇K||²"""
    # Pad kernel to match image size
    target_shape = latent_image.shape
    padded_kernel = jnp.pad(
        kernel,
        ((0, target_shape[0] - kernel.shape[0]),
         (0, target_shape[1] - kernel.shape[1]))
    )
    
    # Main data fitting term in Fourier domain
    kernel_ft = jnp.fft.fft2(padded_kernel)
    latent_ft = jnp.fft.fft2(latent_image)
    convolved = jnp.real(jnp.fft.ifft2(kernel_ft * latent_ft))
    data_term = jnp.sum((convolved - blurred_image) ** 2)
    
    # Simple L2 regularization on kernel
    reg_term = beta * jnp.sum(kernel ** 2)
    
    return data_term + reg_term

def estimate_kernel_jax(current_image, blurred_image, kernel_size, num_iterations=10):
    """Step 2: Kernel Estimation using direct pixel comparison"""
    # Convert to JAX arrays
    current_image = jnp.array(current_image)
    blurred_image = jnp.array(blurred_image)
    
    # Initialize kernel and optimizer
    key = jax.random.PRNGKey(0)
    kernel = jax.random.uniform(key, (kernel_size, kernel_size))
    kernel = kernel / jnp.sum(kernel)
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(kernel)
    grad_fn = jit(grad(energy_function))
    
    for i in range(num_iterations):
        grads = grad_fn(kernel, current_image, blurred_image)
        updates, opt_state = optimizer.update(grads, opt_state)
        kernel = optax.apply_updates(kernel, updates)
        kernel = jnp.maximum(0, kernel)  # ensure non-negativity
        kernel = kernel / jnp.sum(kernel)  # normalize

        logger.log(kernel,"k")
    
    return np.array(kernel)
def deconvolve(blurred, kernel):
    """Step 3: Wiener deconvolution"""
    kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
    blurred_ft = np.fft.fft2(blurred)
    
    noise_level = 0.01
    deblurred_ft = (blurred_ft * np.conj(kernel_ft)) / (np.abs(kernel_ft)**2 + noise_level)
    deblurred = np.real(np.fft.ifft2(deblurred_ft))
    
    return np.clip(deblurred, 0, 1)  # Ensure values are in [0,1]



# Example usage
def test_deblurring():
    # Create a simple test image
    image = cv2.cvtColor(cv2.imread('images/synthetic_deformed/02_4.png'), cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256,256))
    image = image.astype(np.float64)
    image /= 255.0

    # image[25:75, 25:75] = 1  # white square
    
    # Add some synthetic motion blur
    # kernel = np.zeros((25, 25))
    # kernel[12, :] = 1/25  # horizontal motion blur
    # blurred = cv2.filter2D(image, -1, kernel)
    
    # Add some noise
    blurred = image
    blurred += np.random.normal(0, 0.01, blurred.shape)
    blurred = np.clip(blurred, 0, 1)
    
    # Deblur
    deblurred, estimated_kernel = deblur_image(blurred)
    
    return blurred, deblurred, estimated_kernel

blurred, deblurred, estimated_kernel = test_deblurring()


logger.display()
# plt.gray()
# plt.imshow(blurred)
# plt.imshow(deblurred)
