import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, List, Optional, Tuple, Type, Union
from torch import Tensor
import math

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_pretrained_weights(self, weights_path):
        pretrained_dict = torch.load(weights_path)
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which consists of linear layers.
    """

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        Forward pass
        """
        
        if up_x.shape != down_x.shape:
            x = self.upsample(up_x)
            temp_concat = torch.cat([x, down_x], dim=2)
            
            x = torch.cat([temp_concat, temp_concat], dim=3)
        else:
            temp_concat = torch.cat([up_x, down_x], dim=2)
            x = torch.cat([temp_concat, temp_concat], dim=3)
            
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class UNetWithResnet50Encoder(BaseFeaturesExtractor):
    DEPTH = 6
    def __init__(self, observation, output_channels):

        num_cnn_channels = observation['map'].shape[0] + observation['global'].shape[0] + observation['factory'].shape[0] + observation['unit'].shape[0]
        super(UNetWithResnet50Encoder, self).__init__(num_cnn_channels, output_channels)
        
        resnet = ResNet(Bottleneck, [3, 4, 6, 3]).cuda()
        resnet.load_pretrained_weights('net/models/resnet50-DEFAULT.pth')
        down_blocks = []
        up_blocks = []

        num_output_channels = resnet.conv1.out_channels

        new_conv1 = nn.Conv2d(num_cnn_channels, num_output_channels, kernel_size=resnet.conv1.kernel_size,
                      stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)

        resnet.conv1 = new_conv1

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.last_down_block = self.down_blocks[-1]
        self.last_conv2d_layer = list(self.last_down_block.children())[-1]
        self.last_conv2d_layer_out_channels = self.last_conv2d_layer.conv3.out_channels

        self.flatten = nn.Flatten()

        #self.aap2d = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(in_features=8192, out_features=4096)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(in_features=4096, out_features=2048)

        self.lin3 = nn.Linear(in_features=4096, out_features=2048)
        self.lin3 = nn.Linear(in_features=2048, out_features=1024)

        self.lin35 = nn.Linear(in_features=1024, out_features=2048)

        self.lin4 = nn.Linear(in_features=2048, out_features=4096)

        self.lin5 = nn.Linear(in_features=4096, out_features=8192)



        #self.bridge = Bridge(self.last_conv2d_layer_out_channels, self.last_conv2d_layer_out_channels, self.last_conv2d_layer_out_channels * 4)
        #up_blocks.append(UpBlockForUNetWithResNet50(self.last_conv2d_layer_out_channels, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        #up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    #up_conv_in_channels=256, up_conv_out_channels=128))
        #up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + num_cnn_channels, out_channels=64,
                                                    #up_conv_in_channels=128, up_conv_out_channels=64))
        


        self.layers = nn.Sequential(
            # First layer: 12x12 to 24x24, reduce channels from 256 to 128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Activation function can help with non-linear capacity between layers
            # Second layer: 24x24 to 48x48, reduce channels from 128 to 50
            nn.ConvTranspose2d(128, 50, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Final layer: Adjust channels to 25 without changing dimension
            nn.ConvTranspose2d(50, 20, kernel_size=3, stride=1, padding=1),
        )
        self.upconv_1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.transpose = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=2,
            kernel_size=48,
            stride=48,
            padding=0
        )

        self.out = nn.Conv2d(64, output_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        
        cnn_features = x['map']
        global_features = x['global']
        factory_features = x['factory']
        unit_features = x['unit']

        #global_features = self.transpose(global_features.unsqueeze(-1).unsqueeze(-1))


        x = torch.cat((cnn_features, global_features, factory_features, unit_features), dim=1)

        pre_pools = dict()
        pre_pools[f"layer_0"] = x

        x = self.input_block(x)


        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)



        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue

            pre_pools[f"layer_{i}"] = x

        
        
        #last_size = pre_pools[f"layer_{UNetWithResnet50Encoder.DEPTH - 2}"].shape[-1]


        x = self.flatten(x)

        x = self.lin(x)

        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)

        x = self.lin35(x)
        x = self.relu(x)

        x = self.lin4(x)
        x = self.relu(x)
        x = self.lin5(x)
        x = self.relu(x)

        x = x.view(-1, 2048, 2, 2)

        x = F.interpolate(x, size=(3, 3), mode='bilinear', align_corners=False)
        x = self.upconv_1(x)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        

        x = self.layers(x)
        del pre_pools
        return x
    


