import numpy as np
from scipy import fft
import matplotlib.pyplot as plt


def wiener_filter(g, h, n):
    """
    Implements the Wiener filter for image restoration using NumPy's FFT.
    
    Parameters:
    -----------
    g : ndarray
        Input degraded image (2D numpy array, normalized to range [0,1])
    h : ndarray
        Point spread function (2D numpy array, normalized to range [0,1])
    n : ndarray
        Noise image (2D numpy array, normalized to range [0,1])
        
    Returns:
    --------
    f_restored : ndarray
        Restored image (2D numpy array, normalized to range [0,1])
    """

    # Normalize PSF
    h = h / np.sum(h)
    
    # Transform to frequency domain
    G = np.fft.fft2(g)
    H = np.fft.fft2(h, s=g.shape)
    
    # Calculate noise power spectrum
    noise_power = np.mean(np.abs(np.fft.fft2(n))**2)
    
    # Construct Wiener filter
    H_conj = np.conjugate(H)
    H_abs_sq = np.abs(H)**2
    W = H_conj / (H_abs_sq + noise_power + 1e-10)
    
    # Apply filter and transform back to spatial domain
    F_restored = G * W
    f_restored = np.real(np.fft.ifft2(F_restored))
    
    return np.clip(f_restored, 0, 1)

def wiener_deconvolution(blurred_image, kernel, noise_ratio=0.01):
    """
    Perform Wiener deconvolution on a blurred image using NumPy.
    
    Args:
        blurred_image: Array of shape (H, W) - the blurred input image
        kernel: Array of shape (k, k) - the blurring kernel
        noise_ratio: Float - regularization parameter (noise to signal ratio)
    
    Returns:
        Array of shape (H, W) - the deconvolved image
    """
    # Get image dimensions
    H, W = blurred_image.shape
    k_h, k_w = kernel.shape
    
    # Pad kernel to match image size
    padded_kernel = np.zeros((H, W))
    start_h = (H - k_h) // 2
    start_w = (W - k_w) // 2
    padded_kernel[start_h:start_h + k_h, start_w:start_w + k_w] = kernel
    
    # Center and normalize the kernel
    padded_kernel = np.roll(padded_kernel, 
                           shift=(-(k_h//2), -(k_w//2)), 
                           axis=(0, 1))
    padded_kernel = padded_kernel / padded_kernel.sum()
    
    # Convert to frequency domain
    blurred_freq = np.fft.rfft2(blurred_image)
    kernel_freq = np.fft.rfft2(padded_kernel)
    
    # Wiener deconvolution in frequency domain
    # G = H* / (|H|^2 + NSR)
    kernel_conj = np.conj(kernel_freq)
    denominator = np.abs(kernel_freq) ** 2 + noise_ratio
    wiener_filter = kernel_conj / denominator
    
    # Apply filter
    result_freq = blurred_freq * wiener_filter
    
    # Convert back to spatial domain
    result = np.fft.irfft2(result_freq, s=(H, W))
    
    # Handle numerical artifacts
    result = np.clip(result, 0, 1)
    
    return result

def degrade_image(f, h, n):
    """
    Degrades an image by applying blur and noise.
    
    Parameters:
    -----------
    f : ndarray
        Original image (2D numpy array, normalized to range [0,1])
    h : ndarray
        Point spread function (2D numpy array, normalized to range [0,1])
    n : ndarray
        Noise image (2D numpy array, normalized to range [0,1])
        
    Returns:
    --------
    g : ndarray
        Degraded image (2D numpy array, normalized to range [0,1])
    """
    
    # Normalize PSF to sum to 1
    h = h / np.sum(h)
    
    # Apply blur in frequency domain
    F = np.fft.fft2(f)
    H = np.fft.fft2(h, s=f.shape)
    N = np.fft.fft2(n, s=f.shape)


    g = np.real(np.fft.ifft2(F * H + N))
    
    # Add noise to create final degraded image
    
    # Ensure output is in valid range [0,1]
    return np.clip(g, 0, 1)

def show_original_with_ff(k):
    K = np.fft.fftshift(np.fft.fft2(k))

    K_abs =  np.log(np.abs(K) + 1)

    fig, ax = plt.subplots(1,2, figsize=(10,20))

    ax[0].imshow(k)
    ax[1].imshow(K_abs)

def plot_images(images):
    num_images = len(images)

    print(num_images)

    fig, axs = plt.subplots(
        1, num_images,
        figsize=(15 * num_images, 15)
    )
    # fig.gray()

    for idx, img in enumerate(images):
        axs[idx].imshow(img)


import numpy as np

def transform_image(A, H):
    """
    Transform image A using homogeneous transformation matrix H using backward mapping
    
    Args:
        A: Input image (2D numpy array)
        H: 3x3 homogeneous transformation matrix
        
    Returns:
        Transformed image as 2D numpy array
    """
    # Get image dimensions
    height, width = A.shape
    
    # Create output image of same size
    output = np.zeros_like(A)
    
    # Get inverse transformation matrix (for backward mapping)
    H_inv = np.linalg.inv(H)
    
    # Create coordinate matrices
    y, x = np.indices(( width, height))
    
    # Create homogeneous coordinates (x, y, 1)
    coords = np.stack([x, y, np.ones_like(x)], axis=-1)
    
    # Transform coordinates using inverse matrix
    transformed_coords = np.tensordot(coords, H_inv.T, axes=1)
    
    # Convert back from homogeneous coordinates
    transformed_x = transformed_coords[..., 0] / transformed_coords[..., 2]
    transformed_y = transformed_coords[..., 1] / transformed_coords[..., 2]
    
    # Find valid coordinates (within image bounds)
    valid_mask = (
        (transformed_x >= 0) & 
        (transformed_x < width - 1) & 
        (transformed_y >= 0) & 
        (transformed_y < height -1 )
    )
        # Get the four nearest neighbor coordinates for each pixel
    x0 = np.floor(transformed_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(transformed_y).astype(int)
    y1 = y0 + 1
    
    # Calculate interpolation weights
    wx1 = transformed_x - x0
    wx0 = 1 - wx1
    wy1 = transformed_y - y0
    wy0 = 1 - wy1
    
    # Perform bilinear interpolation
    valid_indices = np.where(valid_mask)
    output[x[valid_indices], y[valid_indices]] = (
        wx0[valid_mask] * wy0[valid_mask] * A[x0[valid_mask], y0[valid_mask]] +
        wx1[valid_mask] * wy0[valid_mask] * A[x1[valid_mask], y0[valid_mask]] +
        wx0[valid_mask] * wy1[valid_mask] * A[x0[valid_mask], y1[valid_mask]] +
        wx1[valid_mask] * wy1[valid_mask] * A[x1[valid_mask], y1[valid_mask]]
    )
    
    return output.astype(A.dtype)

def get_rotation_matrix(angle_degrees):
    """
    Get 3x3 homogeneous rotation matrix
    
    Args:
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        3x3 numpy array representing rotation transformation
    """
    # Convert angle to radians
    angle = np.radians(angle_degrees)
    
    # Calculate sine and cosine
    c = np.cos(angle)
    s = np.sin(angle)
    
    # Create rotation matrix
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def get_translation_matrix(tx, ty):
    """
    Get 3x3 homogeneous translation matrix
    
    Args:
        tx: Translation in x direction
        ty: Translation in y direction
    
    Returns:
        3x3 numpy array representing translation transformation
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def get_scale_matrix(sx, sy=None):
    """
    Get 3x3 homogeneous scaling matrix
    
    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction (if None, uses sx for uniform scaling)
    
    Returns:
        3x3 numpy array representing scaling transformation
    """
    # If sy is not provided, use sx for uniform scaling
    if sy is None:
        sy = sx
        
    return np.array([
        [sx,  0, 0],
        [ 0, sy, 0],
        [ 0,  0, 1]
    ])

def get_rotation_matrix_around_point(angle_degrees, cx, cy):
    """
    Get 3x3 homogeneous rotation matrix for rotation around a specific point
    
    Args:
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
        cx, cy: Coordinates of rotation center
    
    Returns:
        3x3 numpy array representing rotation transformation
    """
    # Create basic matrices
    T1 = get_translation_matrix(-cx, -cy)  # Translate to origin
    R = get_rotation_matrix(angle_degrees)  # Rotate
    T2 = get_translation_matrix(cx, cy)     # Translate back
    
    # Combine matrices: T2 @ R @ T1
    return T2 @ R @ T1

