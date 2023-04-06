import torch
import seaborn as sns
import colorcet as cc

class Decompositor(object):
    """ The Decompositor class decomposes the segmented sequences and two image grids that
        containing the exact visual sequences of masks and transformed sprites, selected for the corresponding line
        - mask_sequence: the sequence of masks
        - tsf_proto_sequence : the sequence of the transformed sprites
    """
    def __init__(self, trainer, decompose=True):
        self.model = trainer.model
        self.colors = [c for c in sns.color_palette(cc.glasbey, self.model.sprites.n_sprites) for _ in range(trainer.model.sprites.per_character)]
        self.decompose = decompose

    @torch.no_grad()
    def expand_colors(self, base):
        colors = torch.Tensor(self.colors)
        colors = colors.unsqueeze(-1).unsqueeze(-1)  #size (K,3,1,1)
        colors = colors.expand(len(base), *base[0].size()) #size (K,3,H_sprite,W_sprite)
        return colors.to(base[0].device)

    @torch.no_grad()
    def expand_colors_layers(self, tsf_layers, selection):
        n_cells = len (tsf_layers)
        tsf_layers_colored = list()
        colors = torch.cat([torch.Tensor(self.colors), torch.zeros((1,3))], dim = 0)

        for p in range(n_cells):
            H, W_predicted = tsf_layers[p].size(-2), tsf_layers[p].size(-1)
            colors_p = colors[selection[:,:,p].argmax(0)].unsqueeze(-1).unsqueeze(-1)  #size (N,3,1,1)
            layers_colored = colors_p.expand(colors_p.size(0),*tsf_layers[p][0].size())  #size (N,3,H,W_predicted)
            tsf_layers_colored.append(layers_colored)

        return tsf_layers_colored

    @torch.no_grad()
    def __call__(self, x):
        self.model.eval()

        with torch.no_grad():
            y = self.model.predict_cell_per_cell(x)
            proto = self.model.prototypes.data.clone() # size (K, 3, H_sprite, W_sprite)
            self.model.prototypes.data.copy_(self.expand_colors(proto))

            tsf_layers = y['tsf_layers']

            if not 'selection' in y.keys():
                weights = torch.eye(y['w'].shape[-1]).to(y['w'])[y['w'].argmax(-1)]  #size (N*n_cells,K)
                selection = weights.reshape(len(x),-1,weights.size(-1)).permute(2,0,1)  #size (K,N,n_cells)
            else:
                selection = y['selection']  #size (K,N,n_cells)
                
            tsf_layers_colored = self.expand_colors_layers(tsf_layers, selection)
            r = (x['x'], y['tsf_bkgs'], tsf_layers_colored, y['tsf_masks'])
            y['segmentation'] = self.model.compositor(*r)['cur_img']

            self.model.prototypes.data.copy_(proto)

        return y
