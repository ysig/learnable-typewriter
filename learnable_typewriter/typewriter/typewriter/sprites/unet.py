# Code taken from https://github.com/vye16/deformable-sprites
# Vickie Ye, Zhengqi Li, Richard Tucker, Angjoo Kanazawa and Noah Snavely. "Deformable sprites for unsupervised video decomposition.", CVPR, 2022.
# All rights reserved.

import torch
from torch import nn
import torch.nn.functional as F

def get_norm_cls(norm_fn, **kwargs):
    if norm_fn == "batch":
        return nn.BatchNorm2d
    if norm_fn == "group":

        def get_group_norm(num_channels):
            return nn.GroupNorm(num_channels=num_channels, **kwargs)

        return get_group_norm
    if norm_fn == "instance":
        return nn.InstanceNorm2d
    if norm_fn == "none":

        def get_nop_norm(num_channels):
            return nn.Identity()

        return get_nop_norm
    raise NotImplementedError

def get_nl(nl_fn):
    if nl_fn == "relu":
        return nn.ReLU(inplace=True)
    if nl_fn == "leaky":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if nl_fn == "tanh":
        return nn.Tanh()
    if nl_fn == "sigmoid":
        return nn.Sigmoid()
    if nl_fn == "none" or nl_fn is None:
        return nn.Identity()


def init_linear_block(d_in, d_out, include_nl=True, nl_fn="relu", init_fn=None):
    linear = nn.Linear(d_in, d_out)
    if init_fn is None:
        with torch.no_grad():
            nn.init.kaiming_normal_(linear.weight, a=0, mode="fan_in")
    else:
        init_fn(linear.weight)
    modules = [linear]
    if include_nl:
        modules.append(get_nl(nl_fn))
    return nn.Sequential(*modules)


def init_kaiming(m, nl="relu"):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nl)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_normal(m, mean=0, std=1):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=mean, std=std)
        nn.init.normal_(m.bias, mean=mean, std=std)


def pad_diff(x1, x2):
    diffY = x2.size()[-2] - x1.size()[-2]
    diffX = x2.size()[-1] - x1.size()[-1]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return x1


class ConvBlock(nn.Module):
    """Conv => [BN] => ReLU"""

    def __init__(
        self,
        in_planes,
        planes,
        kernel_size=3,
        padding=1,
        stride=1,
        norm_fn="batch",
        nl_fn="relu",
        use_bias=False,
        **kwargs
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.in_planes = in_planes
        self.planes = planes

        num_groups = planes // 8
        norm_cls = get_norm_cls(norm_fn, num_groups=num_groups)
        self.conv = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=use_bias,
        )
        self.norm = norm_cls(planes)
        self.act = get_nl(nl_fn)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """ConvTranspose => [BN] => ReLU"""

    def __init__(
        self,
        in_planes,
        planes,
        kernel_size=3,
        padding=1,
        stride=1,
        norm_fn="batch",
        nl_fn="relu",
        use_bias=False,
        **kwargs
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.in_planes = in_planes
        self.planes = planes

        num_groups = planes // 8
        norm_cls = get_norm_cls(norm_fn, num_groups=num_groups)
        self.conv = nn.ConvTranspose2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=use_bias,
        )
        self.norm = norm_cls(planes)
        self.act = get_nl(nl_fn)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """(Conv2d => [BN] => ReLU) * 2 with optional skip connection"""

    def __init__(
        self,
        in_planes,
        planes,
        mid_planes=None,
        norm_fn="batch",
        kernel_size=3,
        padding=1,
        stride=1,
        skip=False,
    ):
        super().__init__()
        self.norm_fn = norm_fn
        if mid_planes is None:
            mid_planes = planes
        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.planes = planes

        num_groups = planes // 8
        norm_cls = get_norm_cls(norm_fn, num_groups=num_groups)

        s2 = max(1, stride // 2)
        s1 = max(1, stride // s2)

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=kernel_size, padding=padding, stride=s1
        )
        self.conv2 = nn.Conv2d(
            mid_planes, planes, kernel_size=kernel_size, padding=padding, stride=s2
        )

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_cls(mid_planes)
        self.norm2 = norm_cls(planes)

        self.skip = skip
        if skip and (stride != 1 or in_planes != planes):
            self.out_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)
        else:
            self.out_conv = None

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if not self.skip:
            return y
        if self.out_conv is not None:
            x = self.out_conv(x)
        return x + y


class BottleneckBlock(nn.Module):
    """
    Bottleneck the conv channels from D_in -> D // 4 -> D // 4 -> D,
    with skip connection from input to output, kernel sizes 1 -> 3 -> 1
    """

    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        norm_cls = get_norm_cls(norm_fn, num_groups=num_groups)
        self.norm1 = norm_cls(planes // 4)
        self.norm2 = norm_cls(planes // 4)
        self.norm3 = norm_cls(planes)

        if stride == 1 and in_planes == planes:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.out_conv is not None:
            x = self.out_conv(x)

        return x + y

class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        n_levels=3,
        d_hidden=16,
        fac=2,
        norm_fn="batch",
        init_std=0,
    ):
        super().__init__()
        enc_dims = get_unet_dims(n_levels, d_hidden)

        self.enc = Encoder(n_channels, enc_dims, norm_fn=norm_fn, fac=fac)
        self.dec = Decoder(n_classes, enc_dims, norm_fn=norm_fn, fac=fac)
        self.in_dims = self.enc.in_dims + self.dec.in_dims
        self.out_dims = self.enc.out_dims + self.dec.out_dims

        ## init the last layer
        if init_std > 0:
            # print("init last zero")
            init_normal(self.dec.outc, std=init_std)
        else:
            # print("init last kaiming")
            init_kaiming(self.dec.outc)

    def forward(self, x, idx=None, ret_latents=False, **kwargs):
        z, dn_latents = self.enc(x)
        out, up_latents = self.dec(z, dn_latents, ret_latents=ret_latents)
        if ret_latents:
            dn_latents.append(z)
            return out, dn_latents, up_latents
        return out


class Encoder(nn.Module):
    def __init__(self, n_channels, dims, norm_fn="batch", fac=2):
        super().__init__()
        self.n_channels = n_channels
        self.in_dims, self.out_dims = dims[:-1], dims[1:]

        self.inc = ConvBlock(n_channels, dims[0], norm_fn=norm_fn)

        self.down_layers = nn.ModuleList([])
        for d_in, d_out in zip(self.in_dims, self.out_dims):
            # print("Down layer", d_in, d_out)
            self.down_layers.append(DownSkip(d_in, d_out, norm_fn=norm_fn, fac=fac))

        # print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x):
        x = self.inc(x)
        dn_latents = []
        for layer in self.down_layers:
            dn_latents.append(x)
            x = layer(x)
        return x, dn_latents


class Decoder(nn.Module):
    def __init__(self, n_classes, dims, norm_fn="batch", fac=2):
        super().__init__()
        self.n_classes = n_classes

        dims = dims[::-1]
        d_in = dims[0]
        skip_dims = dims[1:]
        out_dims = []

        self.up_layers = nn.ModuleList([])
        for d_skip in skip_dims:
            d_out = d_skip
            # print("Up layer", d_in, d_out)
            self.up_layers.append(UpSkip(d_in, d_skip, norm_fn=norm_fn, fac=fac))
            d_in = d_out + d_skip
            out_dims.append(d_in)

        self.outc = nn.Conv2d(d_in, n_classes, kernel_size=1)

        self.in_dims = dims[0:1] + out_dims[:-1]
        self.out_dims = out_dims

        # print("INITIALIZING WEIGHTS")
        self.apply(init_kaiming)

    def forward(self, x, dn_latents, ret_latents=False, **kwargs):
        up_latents = []
        for layer, z in zip(self.up_layers, dn_latents[::-1]):
            x = layer(x, z, ret_latents=ret_latents)
            if ret_latents:
                up_latents.append(x)
        out = self.outc(x)
        if ret_latents:
            return out, up_latents
        return out, None


def get_unet_dims(n_levels, d_hidden):
    dims = [d_hidden * 2 ** i for i in range(n_levels)]
    dims.append(dims[-1])
    return dims


class DownSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
            nn.MaxPool2d(fac),
        )

    def forward(self, x):
        return self.block(x)


class UpSkip(nn.Module):
    def __init__(self, d_in, d_out, norm_fn="batch", nl_fn="relu", fac=2):
        super().__init__()

        # use the normal convolutions to reduce the number of channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=fac, mode="bilinear", align_corners=False),
            ConvBlock(
                d_in,
                d_out,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_fn=norm_fn,
                nl_fn=nl_fn,
            ),
        )

    def forward(self, x1, x2, ret_latents=False):
        x1 = self.block(x1)
        x1 = pad_diff(x1, x2)
        x1 = torch.cat([x1, x2], dim=1)
        return x1