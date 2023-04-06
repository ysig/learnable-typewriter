from torch import nn
from learnable_typewriter.typewriter.utils import get_clamp_func
from learnable_typewriter.typewriter.typewriter.sprites.sprites import init_objects

class Background(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.exists = len(cfg) > 0
        if self:
            self.size = cfg['size']
            self.init = cfg['init']

            background = init_objects(1, 3, self.size, self.init)[0]
            self.background_ = nn.Parameter(background)
    
            if self.init['freeze']:
                self.background_.requires_grad = False

            self.clamp_func = get_clamp_func(cfg['use_clamp'])
            
            # For full line of text inference
            self.moving_avg_bkg = True

    def __bool__(self):
        return self.exists

    @property
    def backgrounds(self):
        return self.clamp_func(self.background_)
