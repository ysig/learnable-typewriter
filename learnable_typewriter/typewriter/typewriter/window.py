import torch

class Window(object):
    def __init__(self, encoder, cfg):
        # Image and cell shape
        self.w = cfg['size']
        self.w_step = encoder.feature_downscale_factor * encoder.pool_w
        self.w_extra_oneside_l = self.w//2
        self.w_extra_oneside_r = self.w - self.w_extra_oneside_l

    def global_index(self, p, w_max=None):
        """ Return the index to crop the input image to the p window plus the fact the image is cropped or not """
        crop_on_right, crop_on_left = False, False

        origin = self.w_step * p
        
        raw_x_start = origin - self.w_extra_oneside_l
        if raw_x_start >= 0:
            x_start = raw_x_start
        else:
            crop_on_left = True
            x_start = 0

        raw_x_end = origin + self.w_extra_oneside_r
        if raw_x_end <= w_max:
            x_end = raw_x_end
        else:
            crop_on_right = True
            x_end = w_max

        return x_start, x_end, crop_on_right, crop_on_left

    @torch.no_grad()
    def local_index(self, crop_width, crop_on_left, crop_on_right, offset=0):
        """ Return the index for cropping the sprites """
        offset = offset*self.w_step
        l_x_end = self.w + offset

        l_x_start = 0
        if crop_on_left and not crop_on_right:
            l_x_start = self.w + offset - crop_width

        if crop_on_right:
            l_x_end = crop_width

        return l_x_start, l_x_end

    def get_local_from_global(self, p, w_max):
        ws, we, crop_on_right, crop_on_left = self.global_index(p, w_max)
        lws, lwe = self.local_index(we - ws, crop_on_left, crop_on_right, offset=0)
        return ws, we, lws, lwe
