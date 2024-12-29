import numpy as np
from scipy import signal
import cv2

class BlindDeconvolution:
    def __init__(self, max_kernel_size=(31, 31), num_scales=3):
        self.max_kernel_size = max_kernel_size
        self.num_scales = num_scales
        self.scale_factor = 2
        self.alpha = 100  # Regularization parameter for latent image
        self.beta = 1  # Regularization parameter for kernel
        
    def create_gabor_filters(self, ksize=21):
        """
        Create Gabor filter bank with parameters specified in the paper:
        λ=0.5, ψ=90°, σ=4, γ=1, θ={0°, 60°, 120°}
        """
        filters = []
        theta_values = [0, np.pi/3, 2*np.pi/3]  # 0°, 60°, 120°
        lambd = 0.5
        sigma = 4
        gamma = 1
        psi = np.pi/2  # 90°
        
        for theta in theta_values:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), 
                sigma=sigma, 
                theta=theta, 
                lambd=lambd, 
                gamma=gamma, 
                psi=psi
            )
            filters.append(kernel)
        
        return filters
    
    def apply_gabor_filters(self, image):
        """
        Apply Gabor filter bank to extract edge information
        """
        filters = self.create_gabor_filters()
        responses = []
        
        for kernel in filters:
            filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
            responses.append(filtered)
            
        return np.array(responses)
    
    def create_image_pyramid(self, image):
        """
        Create image pyramid for coarse-to-fine estimation
        """
        pyramid = [image]
        for i in range(self.num_scales - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid[::-1]  # Return from coarse to fine
    
    def estimate_latent_image(self, blurred_image, kernel, mu=1, max_iter=2):
        """
        Estimate latent image using FISTA algorithm (Fast Iterative Shrinkage-Thresholding)
        Based on equations (8) and (9) from the paper.
        
        Args:
            blurred_image: The observed blurry image
            kernel: Current estimate of the blur kernel
            mu: Regularization parameter (α||x||₂)
            max_iter: Maximum number of iterations
            
        Returns:
            Estimated latent (sharp) image
        """
        # Initialize variables
        x_prev = blurred_image.copy()
        z_curr = blurred_image.copy()
        t = 0.001  # Step size
        q_curr = 1
        
        # Get image dimensions
        height, width = blurred_image.shape
        
        # Generate gradient images using Gabor filters
        gradient_images = self.apply_gabor_filters(blurred_image)
        
        for iter in range(max_iter):
            # Store previous estimate
            x_old = x_prev.copy()
            
            # Step 2: Gradient step
            # Compute Kz - g (residual)
            residual = signal.convolve2d(z_curr, kernel, mode='same') - blurred_image
            
            # Compute gradient term k^T * (Kz - g)
            gradient_term = signal.convolve2d(residual, np.flip(kernel), mode='same')
            
            # Update x using soft-thresholding
            x_prev = self._soft_threshold(
                z_curr - t * gradient_term,
                threshold=mu * t
            )
            
            # Step 3: Update q (momentum parameter)
            q_old = q_curr
            q_curr = (1 + np.sqrt(1 + 4 * q_old**2)) / 2
            
            # Step 4: Update z using momentum
            momentum_factor = (q_old - 1) / q_curr
            z_curr = x_prev + momentum_factor * (x_prev - x_old)
            
            # Ensure non-negativity
            z_curr = np.maximum(z_curr, 0)
            
            # Optional: Check convergence
            if np.sum(np.abs(x_prev - x_old)) < 1e-6:
                break
        
        return x_prev
    
    def _soft_threshold(self, x, threshold):
        """
        Soft thresholding operator as defined in equation (10)
        ℰμt(x) = max(|x| - μt, 0)sign(x)
        """
        # Calculate magnitude
        magnitude = np.abs(x)
        
        # Apply soft thresholding
        with np.errstate(divide='ignore', invalid='ignore'):
            # Compute sign(x) * max(|x| - threshold, 0)
            result = np.multiply(
                np.sign(x),
                np.maximum(magnitude - threshold, 0)
            )
            
        return result
    
    def estimate_kernel(self, latent_image, blurred_image, current_kernel):
        """
        Estimate kernel using IRLS algorithm with conjugate gradient
        """
        # Implementation of IRLS with conjugate gradient
        # This is a simplified version - full implementation needed
        pass
    
    def deconvolve(self, blurred_image):
        """
        Main deconvolution process
        """
        # Create image pyramid
        image_pyramid = self.create_image_pyramid(blurred_image)
        
        # Initialize kernel
        kernel = np.random.uniform(0, 1, (5, 5))  # Start with small kernel
        kernel = kernel / np.sum(kernel)  # Normalize
        
        # Process each scale
        for level in range(len(image_pyramid)):
            current_image = image_pyramid[level]
            
            # Extract edge information using Gabor filters
            edge_info = self.apply_gabor_filters(current_image)
            
            # Alternate between estimating latent image and kernel
            for i in range(5):  # Number of iterations
                # Estimate latent image
                latent_image = self.estimate_latent_image(
                    current_image, 
                    kernel
                )
                
                # Estimate kernel
                kernel = self.estimate_kernel(
                    latent_image,
                    current_image,
                    kernel
                )
            
            # Upscale kernel for next level if not at finest level
            if level < len(image_pyramid) - 1:
                kernel = cv2.resize(
                    kernel, 
                    (kernel.shape[0] * 2, kernel.shape[1] * 2),
                    interpolation=cv2.INTER_LINEAR
                )
                kernel = kernel / np.sum(kernel)  # Normalize
        
        return kernel, latent_image