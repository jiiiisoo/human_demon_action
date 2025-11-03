"""
ResNet-based observation encoder for diffusion policy
Following robomimic's VisualCore architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax pooling layer (from robomimic)
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    """
    def __init__(self, input_shape, num_kp=32, temperature=1.0):
        """
        Args:
            input_shape: (C, H, W) shape of input features
            num_kp: number of keypoints
            temperature: softmax temperature
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        
        # 1x1 conv to reduce channels to num_kp
        self.conv = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
        self._num_kp = num_kp
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
        
        # Create meshgrid for spatial coordinates
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
    
    def forward(self, feature):
        """
        Args:
            feature: (B, C, H, W)
        Returns:
            out: (B, num_kp, 2) - keypoint coordinates
        """
        B = feature.shape[0]
        
        # Reduce channels
        feature = self.conv(feature)  # (B, num_kp, H, W)
        
        # Flatten spatial dimensions
        feature = feature.reshape(B, self._num_kp, self._in_h * self._in_w)  # (B, num_kp, H*W)
        
        # Softmax over spatial locations
        attention = F.softmax(feature / self.temperature, dim=-1)  # (B, num_kp, H*W)
        
        # Compute expected coordinates
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)  # (B, num_kp, 1)
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)  # (B, num_kp, 1)
        
        # Concatenate x and y coordinates
        keypoints = torch.cat([expected_x, expected_y], dim=-1)  # (B, num_kp, 2)
        
        return keypoints
    
    def output_shape(self):
        return [self._num_kp, 2]


class ResNetObsEncoder(nn.Module):
    """
    ResNet18-based observation encoder following robomimic's VisualCore
    Architecture: ResNet18 -> SpatialSoftmax -> Flatten -> Linear
    """
    def __init__(
        self,
        image_channels=3,
        image_size=128,
        num_kp=32,
        feature_dim=64,
        pretrained=True,
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_kp = num_kp
        self.feature_dim = feature_dim
        
        # ResNet18 backbone (remove avgpool and fc)
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Get backbone output shape
        # For 128x128 input: (512, 4, 4)
        with torch.no_grad():
            dummy_input = torch.zeros(1, image_channels, image_size, image_size)
            backbone_out = self.backbone(dummy_input)
            backbone_shape = backbone_out.shape[1:]  # (C, H, W)
        
        # SpatialSoftmax pooling (robomimic style)
        self.pool = SpatialSoftmax(
            input_shape=backbone_shape,
            num_kp=num_kp,
            temperature=1.0
        )
        
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear projection to feature_dim
        pool_output_dim = num_kp * 2  # (num_kp, 2) -> num_kp*2
        self.fc = nn.Linear(pool_output_dim, feature_dim)
        
        print(f"ResNet observation encoder initialized (robomimic style)")
        print(f"  backbone: ResNet18")
        print(f"  backbone output shape: {backbone_shape}")
        print(f"  pooling: SpatialSoftmax (num_kp={num_kp})")
        print(f"  feature_dim: {feature_dim}")
        print(f"  output_dim: {feature_dim}")
    
    def forward(self, obs_image):
        """
        Encode a single observation image to visual features
        
        Args:
            obs_image: (B, C, H, W) - single observation image
        
        Returns:
            obs_features: (B, feature_dim)
        """
        # ResNet18 backbone
        features = self.backbone(obs_image)  # (B, 512, 4, 4) for 128x128 input
        
        # SpatialSoftmax pooling
        keypoints = self.pool(features)  # (B, num_kp, 2)
        
        # Flatten keypoints
        features = self.flatten(keypoints)  # (B, num_kp*2)
        
        # Linear projection to feature_dim
        features = self.fc(features)  # (B, feature_dim)
        
        return features
    
    def output_dim(self):
        """Return output dimension"""
        return self.feature_dim


if __name__ == '__main__':
    # Test encoder
    encoder = ResNetObsEncoder(
        image_channels=3,
        image_size=128,
        num_kp=32,
        feature_dim=64,
        pretrained=False,
    )
    
    # Test with dummy input (single image)
    dummy_obs = torch.randn(4, 3, 128, 128)
    features = encoder(dummy_obs)
    print(f"\nInput shape: {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected output dim: {encoder.output_dim()}")

