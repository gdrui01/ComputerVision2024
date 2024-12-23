import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class KernelEstimationNet(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input_size: size of the input image (m x m)
        output_size: size of the kernel to estimate (n x n)
        """
        super(KernelEstimationNet, self).__init__()
        
        # Calculate the flattened sizes
        self.input_flat = input_size * input_size
        self.output_flat = output_size * output_size
        self.output_size = output_size
        
        # Define network architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.input_flat, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, self.output_flat),
            nn.Softmax(dim=1)  # Ensure kernel sums to 1
        )

    def forward(self, x):
        # x shape: (batch_size, m, m)
        batch_size = x.size(0)
        
        # Flatten the input
        x = x.view(batch_size, -1)
        
        # Encode and decode
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Reshape to kernel size
        x = x.view(batch_size, self.output_size, self.output_size)
        return x

def custom_loss(predicted_kernel, target_kernel, original_image, deformed_image):
    """
    Custom loss combining reconstruction loss and kernel similarity
    """
    # Kernel similarity loss (MSE between predicted and target kernels)
    kernel_loss = F.mse_loss(predicted_kernel, target_kernel)
    
    # Reconstruction loss
    # Apply predicted kernel to original image and compare with blurred image
    # predicted_blur = F.conv2d(
    #     original_image.unsqueeze(1),  # Add channel dimension
    #     predicted_kernel.unsqueeze(1).unsqueeze(1),  # Format for conv2d
    #     padding='same'
    # ).squeeze(1)  # Remove channel dimension
    
    # reconstruction_loss = F.mse_loss(predicted_blur, deformed_image)
    
    # Combine losses
    total_loss = kernel_loss# + 0.5 * reconstruction_loss
    return total_loss

def train_model(model, train_loader, num_epochs, device):
    """
    Training loop for the kernel estimation network
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
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
            loss = custom_loss(predicted_kernels, target_kernels, original_images, deformed_images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 1 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.8f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
        scheduler.step(avg_loss)
