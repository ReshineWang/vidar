import torch
import torch.nn as nn

from .resnet import *
from .unet import *


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "unet":
                self.model = Unet(*args, **kwargs)
            case "resnet":
                self.model = ResNet(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
        train_mean = torch.tensor([-0.21462356, 1.08612859, 0.79082996, -0.32984692, 0.06043469, -0.02821355, 0.67331159, 0.23655808, 1.11126196, 0.82665974, -0.34555018, -0.01577058, 0.00672665, 0.67294836])
        train_std = torch.tensor([0.32683188, 0.99261242, 0.78208578, 0.66655099, 0.24993508, 0.57341701, 0.45221126, 0.31501710, 1.01746464, 0.80794698, 0.71788806, 0.26377317, 0.62372464, 0.45181707])
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        outputs = self.resnet_model(torch.tanh(self.mask_model(images)))
        return outputs


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs
