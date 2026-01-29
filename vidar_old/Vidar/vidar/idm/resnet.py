import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
# 如果想用 ResNet50，解开下面的注释
from torchvision.models import resnet50, ResNet50_Weights

class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, output_dim=14, num_frames=3, input_channels=3, *args, **kwargs):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels*num_frames, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2_x = self._make_layer(64, 3, 1)
        self.conv3_x = self._make_layer(128, 4, 2)
        self.conv4_x = self._make_layer(256, 6, 2)
        self.conv5_x = self._make_layer(512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BottleNeckResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):

        if len(x.shape)==5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)  # 将时间维度融合到通道维度

        else:
            pass
        
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class SpatialSoftmax(nn.Module):
    """
    将 Feature Map [B, C, H, W] 转换为关键点坐标 [B, C*2]
    """
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # 1. Softmax 生成概率热力图
        flat = feature_map.view(B, C, -1)
        heatmap = F.softmax(flat / self.temperature.to(device), dim=2).view(B, C, H, W)
        
        # 2. 生成网格 (Meshgrid) [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 3. 计算期望 (加权平均坐标)
        expected_x = torch.sum(heatmap * pos_x, dim=[2, 3]) # [B, C]
        expected_y = torch.sum(heatmap * pos_y, dim=[2, 3]) # [B, C]
        
        # 4. 拼接 x, y
        return torch.cat([expected_x, expected_y], dim=1) # [B, C*2]

class Resnet_plus(nn.Module):
    def __init__(self, num_frames=3, output_dim=14, backbone_type='resnet50', *args, **kwargs):

        super().__init__()
        
        # ----------------------------------------------------------------
        # 1. 加载预训练 Backbone
        # ----------------------------------------------------------------
        print(f"Loading Pretrained {backbone_type}...")
        if backbone_type == 'resnet18':
            # 使用默认的最优 ImageNet 权重
            weights = ResNet18_Weights.DEFAULT 
            self.backbone = resnet18(weights=weights)
            feature_dim = 512 # ResNet18 最后一层通道数
        elif backbone_type == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
            feature_dim = 2048 # ResNet50 最后一层通道数
        else:
            raise ValueError("Only support resnet18 or resnet50")

        # ----------------------------------------------------------------
        # 2. 魔改第一层 (适应多帧输入)
        # ----------------------------------------------------------------
        # 原始输入是 3 通道 (RGB)，现在是 3 * num_frames (比如 9 通道)
        input_channels = 3 * num_frames
        original_conv1 = self.backbone.conv1
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # --- 关键步骤：权重复用初始化 ---
        # 即使通道数变了，我们也希望利用预训练好的 "边缘检测" 能力
        with torch.no_grad():
            # 原始权重形状: [64, 3, 7, 7]
            # 我们在 dim=1 (通道) 维度重复 num_frames 次
            # 并且除以 num_frames，保持激活值的总能量级不变
            self.backbone.conv1.weight[:] = original_conv1.weight.repeat(1, num_frames, 1, 1) / num_frames
        
        print(f"Modified Conv1 to accept {input_channels} channels. Weights initialized from ImageNet.")

        # ----------------------------------------------------------------
        # 3. 砍掉全连接层和 AvgPool
        # ----------------------------------------------------------------
        # 我们只需要卷积特征提取部分
        # torch vision 的 resnet 如果要把最后两层去掉比较麻烦，
        # 我们直接在 forward 里只调用前面的层即可，不需要显式 del
        
        # ----------------------------------------------------------------
        # 4. 空间 Softmax 层
        # ----------------------------------------------------------------
        self.spatial_softmax = SpatialSoftmax()
        
        # ----------------------------------------------------------------
        # 5. 回归头 (Regression Head)
        # ----------------------------------------------------------------
        # 输入维度是: 特征通道数 * 2 (每个通道一个 x 和一个 y)
        self.head_input_dim = feature_dim * 2 
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
        # 初始化 Head 的权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 只初始化我们自己加的 Head，不要动 Backbone 的权重
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, *args, **kwargs):
        """
        Input: [Batch, Frames, 3, H, W]
        """
        if len(x.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions: [B, T, C, H, W]")
        B, T, C, H, W = x.shape
        
        # 1. 通道堆叠 (Channel Stacking)
        # 把时间 T 融合进通道 C
        # [B, 3, 3, H, W] -> [B, 9, H, W]
        x = x.view(B, T * C, H, W)
        
        # 2. Backbone Forward (手动调用各层，跳过 fc)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) 
        # Output shape: [B, 512, H/32, W/32] (对于 ResNet18)
        
        # 3. Spatial Softmax Pooling
        # 把特征图变成坐标
        # [B, 512, H', W'] -> [B, 1024]
        coords = self.spatial_softmax(x)
        
        # 4. Action Prediction
        action = self.mlp_head(coords)
        
        return action

