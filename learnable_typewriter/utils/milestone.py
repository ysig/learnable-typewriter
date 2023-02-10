class Milestone(object):
    def __init__(self, n_iters, events={},  engage_after=None):
        assert 0 < n_iters 
        self.n_iters = n_iters
        self.per_iter = False

        self.reset()
        self.engage_after = engage_after
        self.events = events
        self.counts = {k: 0 for k in events.keys()}

    def get(self):
        flags = {}
        for event, every in self.events.items():
            if every == 0:
                flags[event] = False
                continue

            flags[event] = self.counts[event] >= self.n_iters*every
            if flags[event]:
                self.counts[event] = 0

        return flags

    def update(self):
        self.t += 1
        for k in self.events.keys():
            self.counts[k] += 1

    def reset(self):
        self.engage_after = None
        self.t = 0  
