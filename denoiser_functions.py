import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.signal import windows

def create_gaussian_window(patch_size=256):
    window = windows.gaussian(patch_size, patch_size/6)
    window_2d = np.outer(window, window)
    window_2d = window_2d / window_2d.max()
    return window_2d

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.sigmoid(att)
        return x * att

class EnhancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            SpatialAttention(64)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            SpatialAttention(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            SpatialAttention(256)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),
            SpatialAttention(512)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(1024),
            ResidualBlock(1024),
            SpatialAttention(1024)
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(512),
            SpatialAttention(512)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            SpatialAttention(256)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            SpatialAttention(128)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            SpatialAttention(64)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        out = self.final(d1)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.sigmoid(att)
        return x * att

def create_gaussian_window(patch_size=256):
    window = windows.gaussian(patch_size, patch_size/6)
    window_2d = np.outer(window, window)
    window_2d = window_2d / window_2d.max()
    return window_2d

def create_patches(image_path, patch_size=256, overlap=26, output_dir=None, prefix=''):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    stride = patch_size - overlap
    n_patches_w = (width - overlap) // stride
    n_patches_h = (height - overlap) // stride
    
    patches = []
    patch_positions = []
    
    window = create_gaussian_window(patch_size)
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            left = j * stride
            top = i * stride
            right = left + patch_size
            bottom = top + patch_size
            
            if right > width or bottom > height:
                continue
            
            patch = img.crop((left, top, right, bottom))
            patches.append(patch)
            patch_positions.append((left, top))
    
    return patches, patch_positions, (n_patches_h, n_patches_w), (width, height), window

def reconstruct_image(patches, patch_positions, original_size, window, patch_size=256):
    width, height = original_size
    reconstructed = np.zeros((height, width, 3))
    weights = np.zeros((height, width))
    
    for patch, (left, top) in zip(patches, patch_positions):
        patch_array = np.array(patch)
        
        for c in range(3):
            reconstructed[top:top+patch_size, left:left+patch_size, c] += \
                patch_array[:, :, c] * window
        
        weights[top:top+patch_size, left:left+patch_size] += window
    
    weights = np.maximum(weights, 1e-10)
    for c in range(3):
        reconstructed[:, :, c] /= weights
    
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    reconstructed = Image.fromarray(reconstructed)
    
    return reconstructed
