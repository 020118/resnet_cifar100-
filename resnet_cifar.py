import torch
import torch.nn as nn
from typing import Optional, Union, Any, Callable
from torch import Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._meta import _IMAGENET_CATEGORIES
from functools import partial
from torchvision.transforms._presets import ImageClassification


def conv1by1(in_channel, out_channel, stride):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3by3(in_channel, out_channel, stride, padding, groups):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=padding, groups=groups, stride=stride, bias=False)



class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, mid_channel, stride: int = 1, padding: int = 1, base_norm: int = 64,
                 group: int = 1, downsample: Optional[nn.Module] = None, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(mid_channel * (base_norm / 64) * group)
        self.conv1 = conv1by1(in_channel=in_channel, out_channel=width, stride=stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3by3(width, width, stride=stride, padding=padding, group=group)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1by1(width, self.expansion*base_norm, stride=stride)
        self.bn3 = norm_layer(self.expansion*base_norm)
        self.relu = nn.ReLU(inplace=True)
        self.in_channel = in_channel
        self.stride = stride
        self.downsample = downsample
        self.norm_layer = norm_layer

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, mid_channel, stride: int = 1, padding: int = 1, 
                 base_norm: int = 16, group: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if group != 1 or base_norm != 16:
            raise ValueError("BasicBlock only support group=1, base_norm=16")
        self.conv1 = conv3by3(in_channel, mid_channel, stride=stride, padding=1, groups=group)
        self.bn1 = norm_layer(mid_channel)
        self.conv2 = conv3by3(mid_channel, mid_channel, stride=1, padding=1, groups=group)
        self.bn2 = norm_layer(mid_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


#调整resnet来适应CIFAR-100数据集的训练
class resnet_cifar100(nn.Module):
    def __init__(self, block: type[Union[BasicBlock, BottleNeck]], layers: list[int], groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, 
                 replace_stride_with_dilation: Optional[list[bool]] = None,
                 zero_init_residual: bool = False):
        super(resnet_cifar100, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 2-element tuple, got {replace_stride_with_dilation}"
            )
        self.block = block
        self.norm_layer = norm_layer
        self.dilation = 1
        self.base_norm = 16
        self.groups = groups
        self.layers = layers
        self.in_channel = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.fc = nn.Linear(64*block.expansion, 100)

        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #论文 https://arxiv.org/abs/1706.02677，关于残差支路零初始化的技巧
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck) and m.bn3.weight is None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is None:
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block: type[Union[BottleNeck, BasicBlock]], 
                   plane: int, blocks: int, stride: int = 1, 
                   dilate: bool = False) -> nn.Sequential:
        downsample = None
        block = self.block
        norm_layer = self.norm_layer
        previous_dilation = self.dilation
        layers = []
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channel != plane * block.expansion:
            downsample = nn.Sequential(
                conv1by1(self.in_channel, plane*block.expansion, stride),
                norm_layer(plane*block.expansion)
            )

        layers.append(
            block(self.in_channel, plane, stride, padding=previous_dilation, base_norm=self.base_norm, 
                  group=self.groups, downsample=downsample)
        )
        self.in_channel = plane * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channel, plane, padding=self.dilation, base_norm=self.base_norm, 
                      group=self.groups, norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)


    def forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_impl(x)

def resnet_build_cifar100(block: type[Union[BottleNeck, BasicBlock]], weight: Optional[WeightsEnum], 
                  layers: list[int], progress: bool, **kwargs: Any) -> resnet_cifar100:
    if weight is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weight.meta["categories"]))
    
    model = resnet_cifar100(block, layers, **kwargs)

    if weight is not None:
        model.load_state_dict(weight.get_state_dict(progress=progress, check_hash=True))

    return model

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES
}

'''  手动进行权重的初始化
def resnet56_weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
'''

def resnet56_build(*, weight=None, process: bool=True, **kwargs):
    return resnet_build_cifar100(BasicBlock, weight, [9,9,9], process, **kwargs)


