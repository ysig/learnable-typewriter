import operator
import numpy as np
import torch
from itertools import groupby
from scipy.ndimage import binary_opening

def best_path(s):
    selection = s['selection']
    return torch.argmax(selection, dim=0)

def cer_aggregate(x, k, aggregate, factor):
    if factor == 1 and aggregate:
        x = torch.unique_consecutive(x, dim=-1)

    x = x.tolist()
    if factor > 1:
        x = [(e//factor if e != k else e) for e in x]
        if aggregate:
            x = [e for e, _ in groupby(x)]

    return [e for e in x if e != k]

def inference(model, x, xp=None, aggregate=True):
    if xp is None:
        xp = model.selection_head(x)

    widths, n_cells, k = torch.tensor(x['w']), xp['selection'].size()[-1], xp['selection'].size()[0]-1
    true_widths_pos = model.true_width_pos(x['x'], widths, n_cells)

    xp = best_path(xp)
    xp = [cer_aggregate(xp[i][:true_widths_pos[i]], k, aggregate=aggregate, factor=model.sprites.per_character) for i in range(xp.size()[0])]

    return xp


class InferenceSupervised(object):
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, x, xp=None):
        self.model.eval()
        return inference(self.model, x, xp, aggregate=True)


class InferenceUnsupervised(object):
    def __init__(self, model):
        self.model = model
        self.empty_sprite_id = len(self.model.sprites)

    def check_overlap(self, data, info):
        wsp, wep, lwsp, lwep = self.model.window.get_local_from_global(info['prev_p'], w_max=self.W)
        ws, we, lws, lwe = self.model.window.get_local_from_global(info['p'], w_max=self.W)

        # overlap
        x, xp = torch.zeros(self.H, self.W), torch.zeros(self.H, self.W)

        xp[:, wsp:wep] = data['tsf_masks'][info['prev_p']][info['b'], 0, :, lwsp:lwep]
        x[:, ws:we] = data['tsf_masks'][info['p']][info['b'], 0, :, lws:lwe]

        x, xp = x > 0.22, xp > 0.22
        x = binary_opening(x, structure=np.ones((2,2)))
        xp = binary_opening(xp, structure=np.ones((2,2)))

        (yid, xid), (_, xid_prev) = np.where(x), np.where(xp)

        if not len(xid):
            # no visibility
            return {'non-visible': True, 'borders': None}

        borders, borders_y = (np.min(xid), np.max(xid)), (np.min(yid), np.max(yid))
        if not len(xid_prev):
            return {'next-symbol': True, 'distance': 100, 'borders': borders, 'borders_y': borders_y}

        borders_previous = (np.min(xid_prev), np.max(xid_prev))

        if (x*xp).sum() > 0:
            if abs(borders_previous[0] - borders[0]) < 5 and info['k'] == info['prev_k']:
                return {'non-visible': True, 'borders': None}
            return {'overall': True, 'overlap': True, 'borders': borders, 'borders_y': borders_y}

        if borders[0] >= borders_previous[1]:
            distance = borders[0] - borders_previous[1]
        elif borders_previous[0] >= borders[1]:
            distance = borders_previous[0] - borders[1]
        else:
            distance = 0

        return {'next-symbol': True, 'distance': distance, 'borders': borders, 'borders_y': borders_y}

    def sort(self, x, b, P, true_widths_pos):
        def label(p):
            return np.argmax(x['selection'][b, p].cpu().numpy())

        def get():
            for p in range(P):
                if p < true_widths_pos[b]:                    
                    k = label(p)
                    if k != self.empty_sprite_id:
                        tsf_masks = x['tsf_masks'][p][b]
                        ws, we, lws, lwe = self.model.window.get_local_from_global(p, w_max=self.W)

                        # overlap
                        q = torch.zeros(self.H, self.W)
                        q[:, ws:we] = tsf_masks[0, :, lws:lwe]

                        xid = np.where(q > 0)[-1]
                        if len(xid):
                            yield (p, k, (np.min(xid) + np.max(xid))/2)

        return [(p, k) for (p, k, _) in sorted(get(), key=operator.itemgetter(2))]

    def inference(self, x, y):
        y_size, x_gt = x['w'], x['x']

        B, _, self.H, self.W = y['reconstruction'].size()
        P = len(y['tsf_layers'])

        y['selection'] = y['selection'].permute(1,2,0)  #size (B,P,K)
        seq_labels, tsf_masks = [], []
        for p in range(len(y['tsf_masks'])):
            masks = y['tsf_masks'][p]/y['tsf_masks'][p].max() if y['tsf_masks'][p].max() > 0 else y['tsf_masks'][p] 
            tsf_masks.append(masks)
        y['tsf_masks'] = tsf_masks

        true_widths_pos = self.model.true_width_pos(x_gt, torch.Tensor(y_size), P)
        for b in range(B):
            seq_label, prev, label = [], None, []
            for p, k in self.sort(y, b, P, true_widths_pos):
                if prev is not None:
                    info = {'b': b, 'p': p, 'prev_p': prev[0], 'k': k, 'prev_k': prev[1]}
                    flags = self.check_overlap(y, info)

                    if flags.get('non-visible', False):
                        continue

                    if y_size is not None:
                        if flags['borders'][0] >= y_size[b]:
                            break

                    if not flags.get('overall', False):
                        assert len(label)
                        seq_label += label
                        label = []

                    if flags.get('next-symbol', False):
                        pass

                    if not flags.get('skip', False):
                        if not (flags.get('overlap', False) and len(label) > 1 and (k in label)):
                            label.append(k)
                else:
                    label.append(k)

                prev = (p, k) # is never set back to None

            if len(label):
                seq_label += label

            seq_labels.append(seq_label)
        return seq_labels

    @torch.no_grad()
    def __call__(self, x, xp=None):
        self.model.eval()
        if xp is None:
            xp = self.model.predict_cell_per_cell(x)
        return self.inference(x, xp)
