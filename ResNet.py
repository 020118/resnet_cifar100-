import torch
import torch.nn as nn
from typing import Optional, Union, Any
from torch import Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._meta import _IMAGENET_CATEGORIES
from functools import partial
from torchvision.transforms._presets import ImageClassification


def conv1by1(in_channel, out_channel, stride, padding):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=padding, stride=stride, bias=False)

def conv3by3(in_channel, out_channel, stride, padding, group):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=padding, groups=group, stride=stride, bias=False)



class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, mid_channel, stride: int = 1, padding: int = 1, base_norm: int = 64,
                 group: int = 1, downsample: Optional[nn.Module] = None, 
                 norm_layer: Optional[callable[..., nn.Module]] = None):
        #middle_channel为当前block的基本通道数，conv2_x为64， conv3_x为128
        #base_norm为基础宽度，用于构建bottelneck的中间层，论文中默认为64, group为bottleneck组数
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = mid_channel * (base_norm / 64) * group
        self.conv1 = conv1by1(in_channel=in_channel, out_channel=width, padding=padding, stride=stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3by3(width, width, stride=stride, padding=padding, group=group)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1by1(width, self.expansion*base_norm, stride=stride, padding=padding)
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
            identity = self.downsample(x)
        
        x += identity
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, mid_channel, stride: int = 1, padding: int = 1, 
                 base_norm: int = 64, group: int = 1,
                 norm_layer: Optional[callable[..., nn.Module]] = None,
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = mid_channel * (base_norm / 64) * group
        self.conv1 = conv3by3(in_channel, width, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3by3(width, width, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(width)
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
            identity = self.downsample(x)

        x += identity
        x = self.relu(x)
        return x

class resnet(nn.Module):
    def __init__(self, block: type[Union[BasicBlock, BottleNeck]], layers: list[int], groups: int = 1,
                 norm_layer: Optional[callable[..., nn.Module]] = None, 
                 replace_stride_with_dilation: Optional[list[bool]] = None):
        super(resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.block = block
        self.norm_layer = norm_layer
        self.in_channel = 64
        self.dilation = 1
        self.base_norm = 64
        self.groups = groups
        self.layers = layers

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
                norm_layer(plane*block)
            )

        layers.append(
            block(self.in_channel, plane, stride, padding=previous_dilation, base_norm=self.base_norm, 
                  group=self.groups, downsample=downsample)
        )
        
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

        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.Linear(2048, 1000)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_impl(x)

def resnet_build(block: type[Union[BottleNeck, BasicBlock]], weight: Optional[WeightsEnum], 
                  layers: list[int], progress: bool, **kwargs: Any) -> resnet:
    if weight is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weight.meta["categories"]))
    
    model = resnet(block, layers, **kwargs)

    if weight is not None:
        model.load_state_dict(weight.get_state_dict(progress=progress, check_hash=True))

    return model



_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES
}


class resnet18_weight(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class resnet34_weight(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 21797672,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_ops": 3.664,
            "_file_size": 83.275,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class resnet50_weight(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

class resnet101_weight(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
            },
            "_ops": 7.801,
            "_file_size": 170.511,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
            },
            "_ops": 7.801,
            "_file_size": 170.53,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

class resnet152_weight(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
            },
            "_ops": 11.514,
            "_file_size": 230.434,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
            },
            "_ops": 11.514,
            "_file_size": 230.474,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_ops": 4.23,
            "_file_size": 95.789,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_ops": 4.23,
            "_file_size": 95.833,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
            },
            "_ops": 16.414,
            "_file_size": 339.586,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }
            },
            "_ops": 16.414,
            "_file_size": 339.673,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 83455272,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }
            },
            "_ops": 15.46,
            "_file_size": 319.318,
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_ops": 11.398,
            "_file_size": 131.82,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_ops": 11.398,
            "_file_size": 263.124,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

class Wide_ResNet101_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
            },
            "_ops": 22.753,
            "_file_size": 242.896,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }
            },
            "_ops": 22.753,
            "_file_size": 484.747,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2



@register_model()  #将resnet18注册到模型注册表中，方便统一管理接口等，如方便后续使用函数get_model(), list_model()
@handle_legacy_interface(weights=("pretrained", resnet18_weight.IMAGENET1K_V1))  #兼容旧版本代码使用pretrained=True
def resnet18(*, weight: Optional[resnet18_weight]=None, progress: bool=True, **kwargs: Any) -> resnet:
    weight = resnet18_weight.verify(weight)

    return resnet_build(BasicBlock, weight, [2,2,2,2], progress, **kwargs)

@register_model()  
@handle_legacy_interface(weights=("pretrained", resnet34_weight.IMAGENET1K_V1))  
def resnet34(*, weight: Optional[resnet34_weight]=None, progress: bool=True, **kwargs: Any) -> resnet:
    weight = resnet34_weight.verify(weight)

    return resnet_build(BasicBlock, weight, [3,4,6,3], progress, **kwargs)

@register_model()  
@handle_legacy_interface(weights=("pretrained", resnet50_weight.IMAGENET1K_V2))  
def resnet50(*, weight: Optional[resnet50_weight]=None, progress: bool=True, **kwargs: Any) -> resnet:
    weight = resnet50_weight.verify(weight)

    return resnet_build(BottleNeck, weight, [3,4,6,3], progress, **kwargs)

@register_model()  
@handle_legacy_interface(weights=("pretrained", resnet101_weight.IMAGENET1K_V2))  
def resnet101(*, weight: Optional[resnet101_weight]=None, progress: bool=True, **kwargs: Any) -> resnet:
    weight = resnet101_weight.verify(weight)

    return resnet_build(BottleNeck, weight, [3,4,23,3], progress, **kwargs)

@register_model()  
@handle_legacy_interface(weights=("pretrained", resnet152_weight.IMAGENET1K_V2))  
def resnet152(*, weight: Optional[resnet152_weight]=None, progress: bool=True, **kwargs: Any) -> resnet:
    weight = resnet152_weight.verify(weight)

    return resnet_build(BottleNeck, weight, [3,8,36,3], progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V2))  
#分组卷积的方式，在论文《Aggregated Residual Transformations for Deep Neural Networks》中提及
#https://arxiv.org/abs/1611.05431
def resnext50_32x4d(*, weight: Optional[ResNeXt50_32X4D_Weights]=None, progress: bool=True, **kwargs: Any) ->resnet:
    weight = ResNeXt50_32X4D_Weights.verify(weight)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "wide_per_group", 4)

    return resnet_build(BottleNeck, weight, [3,4,6,3], progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V2))  
def resnext101_32x8d(*, weight: Optional[ResNeXt101_32X8D_Weights]=None, progress: bool=True, **kwargs: Any) ->resnet:
    weight = ResNeXt101_32X8D_Weights.verify(weight)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "wide_per_group", 8)

    return resnet_build(BottleNeck, weight, [3,4,23,3], progress, **kwargs)

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt101_64X4D_Weights.IMAGENET1K_V1))  
def resnext101_64x4d(*, weight: Optional[ResNeXt101_64X4D_Weights]=None, progress: bool=True, **kwargs: Any) ->resnet:
    weight = ResNeXt101_64X4D_Weights.verify(weight)
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "wide_per_group", 4)

    return resnet_build(BottleNeck, weight, [3,4,23,3], progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> resnet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return resnet_build(BottleNeck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> resnet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return resnet_build(BottleNeck, [3, 4, 23, 3], weights, progress, **kwargs)
