import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

# Параметрами блока будут:
# - количество каналов на входе
# - количество каналов на выходе
# - глубина блока (2 или 3, по количеству конволюционных слоев)
# - kernel_size и padding
#
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size = 3, padding = 1):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))

        for i in range(depth-1):
            self.layers.append(nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #добавляем MaxPool с индексами для последующего Unpooling

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x, indices = self.maxpooling(x)
        return x, indices


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size = 3, padding = 1, tail_activation = True, reduce_channels_at='tail'):
        super(DecoderBlock, self).__init__() 
        
        if reduce_channels_at not in ('head', 'tail'):
            raise ValueError(f"reduce_channels_at should be 'head' or 'tail', got: {reduce_channels_at}")
        
        self.upsampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.layers = nn.ModuleList()

        curr_channels = in_channels
        if reduce_channels_at == 'head':  # задел под переиспользование блоков в UNet
            self.layers.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding))
            curr_channels = out_channels
            self.layers.append(nn.BatchNorm2d(curr_channels))
            self.layers.append(nn.ReLU(inplace=True))

        for i in range(depth-1):
            self.layers.append(nn.Conv2d(in_channels = curr_channels, out_channels = curr_channels, kernel_size = kernel_size, padding = padding))
            self.layers.append(nn.BatchNorm2d(curr_channels))
            self.layers.append(nn.ReLU(inplace=True))

        if reduce_channels_at == 'tail':
            self.layers.append(nn.Conv2d(in_channels = curr_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))

        if not tail_activation:
            del self.layers[-2:]

    def forward(self, x, max_indices):
        x = self.upsampling(x, max_indices)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    

class UEncoderBlock(EncoderBlock):
    def __init__(self, in_channels, out_channels, depth, kernel_size = 3, padding = 1, bottleneck = False):
        super(UEncoderBlock, self).__init__(in_channels, out_channels, depth, kernel_size, padding)

        if bottleneck:
            self.maxpooling = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x_skip = x

        if self.maxpooling is None:
            return x, None, None  # no pooling in bottleneck 

        x, indices = self.maxpooling(x)
        return x, x_skip, indices


class UDecoderBlock(DecoderBlock):

    def __init__(self, in_channels, out_channels, depth, kernel_size = 3, padding = 1, tail_activation = True, reduce_channels_at='head'):
        super(UDecoderBlock, self).__init__(in_channels, out_channels, depth, kernel_size, padding, tail_activation, reduce_channels_at) 

    def forward(self, x, x_skip, max_indices):
        x = self.upsampling(x, max_indices)
        x = torch.cat([x, x_skip], dim=1)  # add skip connections

        for layer in self.layers:
            x = layer(x)
        
        return x


class UNet(nn.Module):
    def __init__(self,n_class=1, in_channels=3, num_features = 64):
        super(UNet, self).__init__()

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=1)

        self.encoders = nn.ModuleList([
            UEncoderBlock(num_features,     num_features * 2,  depth=2),
            UEncoderBlock(num_features * 2, num_features * 4,  depth=2),
            UEncoderBlock(num_features * 4, num_features * 8,  depth=3),
            UEncoderBlock(num_features * 8, num_features * 16, depth=3),
        ])

        self.bottleneck = UEncoderBlock(num_features * 16, num_features * 16, depth=3, bottleneck=True)

        self.decoders = nn.ModuleList([            
            UDecoderBlock(num_features * (16*2), num_features * 8, depth=3),
            UDecoderBlock(num_features * (8*2),  num_features * 4,  depth=3),
            UDecoderBlock(num_features * (4*2),  num_features * 2,  depth=2),
            UDecoderBlock(num_features * (2*2),  num_features,      depth=2),
        ])

        self.out_conv = nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1)


    def forward(self, x):
        x = self.in_conv(x)

        max_pooling_indices = []
        x_skips = []
        for encoder in self.encoders:
            x, x_skip, indices = encoder(x)
            x_skips.append(x_skip)
            max_pooling_indices.append(indices)

        x, _, _ = self.bottleneck(x)

        for decoder, indices, skip in zip(self.decoders, reversed(max_pooling_indices), reversed(x_skips)):
            x = decoder(x, x_skip=skip, max_indices=indices)

        return self.out_conv(x)
    

class UpscalerNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, scale=2):
        super(UpscalerNet, self).__init__()
        
        self.scale = scale

        self.unet = UNet(n_class=out_channels, in_channels=in_channels, num_features=num_features)
        self.final = nn.Conv2d(out_channels, out_channels*scale**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)


    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)

        x = self.unet(x)  # (B, C*S*S, H, W)
        x = self.final(x)  
        x = self.pixel_shuffle(x) # (B, C, H*S, W*S)
        x = F.sigmoid(x)

        return base + x