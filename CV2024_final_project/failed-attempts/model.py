import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class KernelEstimationNetwork(nn.Module):
    def __init__(self, kernel_size=13, hidden_dim=4):
        super(KernelEstimationNetwork, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Encoder network designed for 512x512 input
        # After 5 max pooling layers: 512->256->128->64->32->16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),      # 512x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 256x256
            
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 128x128
            
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 64x64
            
            nn.Conv2d(hidden_dim*4, hidden_dim*8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 32x32
            
            nn.Conv2d(hidden_dim*8, hidden_dim*16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 16x16
            
            nn.Conv2d(hidden_dim*16, hidden_dim*16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)                      # Global average pooling to 1x1
        )
        # MLP to predict kernel weights
        self.kernel_predictor = nn.Sequential(
            nn.Linear(hidden_dim*16, hidden_dim*8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*8, kernel_size * kernel_size)
        )
        
    def forward(self, x):
        # Input size assertion
        assert x.size(2) == 512 and x.size(3) == 512, f"Input spatial dimensions must be 512x512, got {x.size(2)}x{x.size(3)}"
        batch_size = x.size(0)
        
        # Extract global features
        features = self.encoder(x)
        features = features.view(batch_size, -1)
        
        # Predict kernel weights
        kernel_weights = self.kernel_predictor(features)
        
        # Reshape kernel weights to proper format (B, K, K)
        kernel_weights = kernel_weights.view(batch_size, self.kernel_size, self.kernel_size)
        
        # Normalize kernel weights to sum to 1
        kernel_weights = F.softmax(kernel_weights.view(batch_size, -1), dim=1)
        kernel_weights = kernel_weights.view(batch_size, self.kernel_size, self.kernel_size)

        # print("kernel_weights", kernel_weights.shape)
        
        return kernel_weights


def custom_loss(predicted_kernel, target_kernel, original_image, deformed_image):
    """
    Custom loss combining reconstruction loss and kernel similarity.
    Args:
        predicted_kernel: shape (batch_size, kernel_size, kernel_size)
        target_kernel: shape (batch_size, kernel_size, kernel_size)
        original_image: shape (batch_size, 1, height, width)
        deformed_image: shape (batch_size, 1, height, width)
    """
    batch_size = original_image.size(0)
    device = original_image.device
    
    # Initialize losses
    total_kernel_loss = torch.tensor(0.0, device=device)
    total_reconstruction_loss = torch.tensor(0.0, device=device)
    
    # Process each image in the batch separately
    for i in range(batch_size):
        # Get single image and its corresponding kernels
        single_image = original_image[i:i+1]  # Keep dimension: (1, 1, height, width)
        single_deformed = deformed_image[i:i+1]  # (1, 1, height, width)
        single_pred_kernel = predicted_kernel[i]  # (kernel_size, kernel_size)
        single_target_kernel = target_kernel[i]  # (kernel_size, kernel_size)
        
        # Kernel similarity loss for this image
        kernel_loss = F.mse_loss(single_pred_kernel, single_target_kernel)
        total_kernel_loss += kernel_loss
        
        # Reconstruction loss for this image
        # Reshape kernel for convolution: (1, 1, kernel_size, kernel_size)
        conv_kernel = single_pred_kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        predicted_blur = F.conv2d(
            single_image,
            conv_kernel,
            padding='same'
        )
        
        # Compute reconstruction loss for this image
        reconstruction_loss = F.mse_loss(predicted_blur, single_deformed)
        total_reconstruction_loss += reconstruction_loss
    
    # Average losses over batch
    avg_kernel_loss = total_kernel_loss / batch_size
    avg_reconstruction_loss = total_reconstruction_loss / batch_size
    
    # Combine losses
    total_loss = avg_kernel_loss + 0.5 * avg_reconstruction_loss
    
    return total_loss


def train_model(model, train_loader, num_epochs, device):
    """
    Training loop for the kernel estimation network
    """
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (original_images,deformed_images, target_kernels) in enumerate(train_loader):
            # print(blurred_images) 
            deformed_images = deformed_images.to(device)
            
            target_kernels = target_kernels.to(device)
            
            optimizer.zero_grad()
            
            
            # Forward pass 
            predicted_kernels = model(deformed_images)
            
            # Calculate loss

            # print('predicted kernels', predicted_kernels.shape)
            # print('target kernels', target_kernels.shape)
            # print('original_images', original_images.shape)
            # print('deformed_images', deformed_images.shape)

            loss = custom_loss(predicted_kernels, target_kernels, original_images, deformed_images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # total_loss += loss.item()
            
            # if batch_idx % 1 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.12f}')
        
        # avg_loss = total_loss / len(train_loader)
        # print(f'Epoch {epoch} Average Loss: {avg_loss:.12f}, Log loss: {np.log10(avg_loss):.3f}')
        # scheduler.step(avg_loss)
