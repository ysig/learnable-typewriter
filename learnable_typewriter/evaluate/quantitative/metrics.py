from collections import defaultdict

def convert(input):
    multi = 1
    base = 0
    for elt in input[::-1]:
        if not elt == 10:
            base += elt * multi
            multi *= 10
    return(base)

class Metrics(object):
    def __init__(self, *names, sum_for_prop=False):
        self.names = list(names)
        self.curves = defaultdict(list)
        self.meters = defaultdict(AverageMeter)
        self.sum_for_prop = sum_for_prop

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self.meters[name].reset()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ' | '.join(name + ": " + str(self.meters[name].avg) for name in self.names if not name.startswith('prop_clus'))

    @property
    def avg_values(self):
        return [self.meters[name].avg for name in self.names]

    @property
    def sum_values(self):
        return [self.meters[name].sum for name in self.names]

    @property
    def collapse_values(self):
        if not self.sum_for_prop:
            return self.avg_values
        else:
            list_ = []
            for name in self.names:
                if self.sum_for_prop and 'prop_clus' in name:
                    elt = self.meters[name].sum
                else:
                    elt = self.meters[name].avg
                list_ += [elt]
        return list_

    def update(self, *name_val):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v)
        else:
            name, val = name_val
            if name not in self.names:
                self.names.append(name)
            if isinstance(val, (tuple, list)):
                assert len(val) == 2
                self.meters[name].update(val[0], n=val[1])
            else:
                self.meters[name].update(val)

class AverageMeter(object):
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class AverageTensorMeter(object):
    """AverageMeter for tensors of size (B, *dim) over B dimension"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.count = 0

    def update(self, t):
        n = t.size(0)
        if n > 0:
            avg = t.mean(dim=0)
            self.avg = (self.count * self.avg + n * avg) / (self.count + n)
            self.count += n
