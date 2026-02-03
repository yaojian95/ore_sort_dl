import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        features = []
        for size in self.pool_sizes:
            # pool: (B, C, H, W) -> (B, C, size, size)
            # We want to use adaptive pooling to handle any input size
            pool = F.adaptive_avg_pool2d(x, output_size=(size, size))
            
            # Flatten: (B, C, size, size) -> (B, C * size * size)
            flat = pool.view(x.size(0), -1)
            features.append(flat)
            
        # Concatenate all features
        return torch.cat(features, dim=1)

class OreGradeRegressor_ResNet_Spatial(nn.Module):
    def __init__(self, in_channels=2, num_targets=4, pretrained=True):
        super(OreGradeRegressor_ResNet_Spatial, self).__init__()
        
        # Load ResNet18 Backbone
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        
        # We need the full resnet but without the FC layer AND AVGPOOL layer
        original_resnet = resnet18(weights=weights)
        
        # Modify first layer (same as before)
        self.conv1 = nn.Conv2d(
            in_channels, 
            original_resnet.conv1.out_channels, 
            kernel_size=original_resnet.conv1.kernel_size, 
            stride=original_resnet.conv1.stride, 
            padding=original_resnet.conv1.padding, 
            bias=original_resnet.conv1.bias
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        
        # Spatial Pyramid Pooling
        # ResNet18 last feature map has 512 channels
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 2, 4])
        
        # Calculate input features for FC
        # 1x1 + 2x2 + 4x4 = 1 + 4 + 16 = 21 regions
        # Total features = 512 * 21 = 10752
        num_spp_features = 512 * (1*1 + 2*2 + 4*4)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_spp_features, num_targets)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x shape: (B, 512, H/32, W/32) e.g., (B, 512, 4, 4) for 128x128 input
        
        x = self.spp(x)
        # x shape: (B, 10752)
        
        x = self.fc(x)
        return x

class OreGradeRegressor_ResNet(nn.Module):
    def __init__(self, in_channels=2, num_targets=4, pretrained=True):
        super(OreGradeRegressor_ResNet, self).__init__()
        
        # Load ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = resnet18(weights=weights)
        
        # Modify first layer to accept 'in_channels'
        original_first_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            original_first_layer.out_channels, 
            kernel_size=original_first_layer.kernel_size, 
            stride=original_first_layer.stride, 
            padding=original_first_layer.padding, 
            bias=original_first_layer.bias
        )
        
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Modify the fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_targets)
        )

    def forward(self, x):
        return self.backbone(x)

class OreGradeRegressor_MobileNet(nn.Module):
    def __init__(self, in_channels=2, num_targets=4, pretrained=True):
        super(OreGradeRegressor_MobileNet, self).__init__()
        
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = mobilenet_v3_small(weights=weights)
        
        # Modify first layer (MobileNetV3 first layer is named 'features[0][0]')
        # It's a Conv2dNormActivation, so features[0][0] is the Conv2d
        original_first_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        nn.init.kaiming_normal_(self.backbone.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        
        # Modify classifier
        # MobileNetV3 classifier is: Linear -> Hardswish -> Dropout -> Linear
        # We replace the last Linear
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(num_ftrs, num_targets)

    def forward(self, x):
        return self.backbone(x)

class OreGradeRegressor_EfficientNet(nn.Module):
    def __init__(self, in_channels=2, num_targets=4, pretrained=True):
        super(OreGradeRegressor_EfficientNet, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = efficientnet_b0(weights=weights)
        
        # Modify first layer: features[0][0] is Conv2d
        original_first_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        nn.init.kaiming_normal_(self.backbone.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        
        # Modify classifier
        # EfficientNet classifier is: Dropout -> Linear
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, num_targets)

    def forward(self, x):
        return self.backbone(x)

def get_model(model_name, in_channels=2, num_targets=4, pretrained=True):
    if model_name == 'resnet18':
        return OreGradeRegressor_ResNet(in_channels, num_targets, pretrained)
    elif model_name == 'mobilenet_v3_small':
        return OreGradeRegressor_MobileNet(in_channels, num_targets, pretrained)
    elif model_name == 'efficientnet_b0':
        return OreGradeRegressor_EfficientNet(in_channels, num_targets, pretrained)
    elif model_name == 'resnet18_spatial':
        return OreGradeRegressor_ResNet_Spatial(in_channels, num_targets, pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Backward compatibility alias
OreGradeRegressor = OreGradeRegressor_ResNet

if __name__ == "__main__":
    # Test
    models_to_test = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0', 'resnet18_spatial']
    dummy_input = torch.randn(2, 2, 128, 128)
    
    for m_name in models_to_test:
        print(f"Testing {m_name}...")
        model = get_model(m_name, in_channels=2, num_targets=4)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
