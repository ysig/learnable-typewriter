import random
from torch import zeros

class LayeredCompositor(object):
    def __init__(self, model):
        self.model = model
        self.window = model.window

    def init(self):
        size = self.size
        B, C, H, W = size
        if not hasattr(self, 'cur_img'):
            self.cur_mask = zeros(self.n_cells, B, 1, H, W, device=self.device).detach()
            self.cur_foreground = zeros(self.n_cells, B, C, H, W, device=self.device).detach()
            self.cur_img = zeros(size, device=self.device).detach()  # The output / predicted image

            return True
        else:
            return False

    def pop(self):
        [delattr(self, k) for k in ['layers', 'masks', 'background', 'B', 'W', 'K', 'n_cells', 'device', 'size']]
        return {k: self.__dict__.pop(k) for k in ['cur_img', 'cur_mask', 'cur_foreground']}

    def set(self, x, background, layers, masks):
        self.B = x.size()[0]
        self.W = x.size()[-1]
        self.K = self.model.n_prototypes
        self.n_cells = len(layers)
        self.size = x.size()

        self.layers, self.masks, self.background = layers, masks, background
        self.device = x.device

    def get_local_index(self, p, w=None):
        w = (self.W if w is None else w)
        ws, we, crop_on_right, crop_on_left = self.window.global_index(p, w_max=w)
        lws, lwe = self.window.local_index(we - ws, crop_on_left, crop_on_right)
        return ws, we, lws, lwe

    def get_local(self, p):
        output = {}
        ws, we, lws, lwe = self.get_local_index(p)

        output['bounds'] = (ws, we)
        output['layer'] = self.layers[p][:, :, :, lws:lwe]
        output['mask'] = self.masks[p][:, :, :, lws:lwe]
        return output

    def update(self, p):
        # update at position p with selection
        local = self.get_local(p)
        ws, we = local['bounds']
        self.cur_mask[p, :, :, :, ws:we] = local['mask']
        self.cur_foreground[p, :, :, :, ws:we] = local['layer']

    def __call__(self, *input):
        self.set(*input)
        self.init()

        order = list(range(self.n_cells))
        for p in order:
            self.update(p)
        if self.model.training:
            random.shuffle(order)

        self.cur_img = self.cur_img + self.background
        for p in order:
            self.cur_img = self.cur_mask[p]*self.cur_foreground[p] + (1-self.cur_mask[p])*self.cur_img
        
        return self.pop()