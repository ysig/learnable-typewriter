import numpy as np
import torch
from torch import nn, zeros
from torch.nn import functional as F
from learnable_typewriter.typewriter.typewriter.mini_resnet import get_resnet_model

DOWNSCALE_FACTOR = {'resnet32': 4, 'resnet20': 4, 'resnet14': 4, 'resnet8': 4, 'default': 5} 


def gaussian_kernel(size, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    ax = np.linspace(-(size[1] - 1)/ 2., (size[1]-1)/2., size[1])
    ay = np.linspace(-(size[0] - 1)/ 2., (size[0]-1)/2., size[0])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)

    return torch.as_tensor(kernel, dtype=dtype).unsqueeze(0).unsqueeze(0)

class GaussianPool(nn.Module):
    def __init__(self, size, stride):
        super().__init__()
        self.kernel = gaussian_kernel(size)
        self.stride = stride

    def forward(self, x):
        self.kernel = self.kernel.to(x.device)
        kernel = self.kernel.expand(x.size()[1], -1, -1, -1)
        return F.conv2d(x, weight=kernel, stride=self.stride, padding=0, groups=x.size()[1])

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.C, self.H = cfg['C'], cfg['H']

        resnet = get_resnet_model(cfg['name'])(in_channels=self.C)
        seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.layer1, resnet.layer2, resnet.layer3]

        # The resnet32 layer3 divide each dimension by 4 => 50 for an image in [40, 200]
        self.feature_downscale_factor = DOWNSCALE_FACTOR[cfg['name']]
        self.pool_h = self.H // self.feature_downscale_factor
        self.pool_w = cfg['pooling_on_width']

        seq.append(GaussianPool((self.pool_h, self.pool_w), stride=(1, self.pool_w)))

        self.encoder = nn.Sequential(*seq)
        self.out_ch = self.encoder(zeros(1, 3, self.H, 2*self.H)).size()[1]
        self.layer_norm = nn.LayerNorm(self.out_ch, elementwise_affine=False)

        rnn_cfg = cfg.get('rnn', {})
        if len(rnn_cfg):
          type = rnn_cfg.get('type', 'gru')
          assert type in {'gru', 'lstm'}
          rnn_model = (nn.GRU if type == 'gru' else nn.LSTM)
          dropout, bias = rnn_cfg.get('dropout', 0), rnn_cfg.get('bias', True)
          bidirectional = rnn_cfg.get('bidirectional', True)
          self.rnn = rnn_model(self.out_ch, self.out_ch, rnn_cfg.get('num_layers', 3), bidirectional=bidirectional, dropout=dropout, batch_first=True, bias=bias)
          if bidirectional:
              self.out_ch = 2*self.out_ch
          self.infer_rnn_= self.infer_rnn
        else:
          self.infer_rnn_ = nn.Identity()

    def infer_rnn(self, x):
        return self.rnn(x)[0]

    def forward(self, x):
        x = self.encoder(x)
        x = self.layer_norm(x.squeeze(2).permute(0, 2, 1))
        x = self.infer_rnn_(x).permute(0, 2, 1)
        return x
