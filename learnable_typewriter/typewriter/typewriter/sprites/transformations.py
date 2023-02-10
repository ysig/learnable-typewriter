import torch
from torch import nn, stack
from learnable_typewriter.typewriter.typewriter.sprites.transformation_module import IdentityModule, ColorModule, PositionModule

class Transformation(nn.Module):
    def __init_model_pass__(self, model, background):
        tsf_config = {
            'in_channels': model.encoder.out_ch,
            'canvas_size': model.canvas_size,
        }

        if background:
            tsf_config['sprite_size'] = model.background.size
        else:
            tsf_config['sprite_size'] = model.sprites.sprite_size
            tsf_config['canvas_size'] = (model.canvas_size[0], model.window.w)

        return tsf_config

    def __init__(self, model, n, cfg, background=False):
        super().__init__()

        # Input related init
        model_params = self.__init_model_pass__(model, background=background)
        self.tsf_sequences = nn.ModuleList([TransformationChain(model_params, cfg) for _ in range(n)])
        self.is_identity = all(tsf.is_identity for tsf in self.tsf_sequences)

    def __getitem__(self, i):
        return self.tsf_sequences[i]

    def __iter__(self):
        for tsf in self.tsf_sequences:
            yield tsf

    def predict_parameters(self, x=None, features=None):
        if self.is_identity:
            return None
        else:
            features = self.encoder(x) if features is None else features
            return stack([tsf.predict_parameters(features) for tsf in self.tsf_sequences], dim=0)

    def apply_parameters(self, prototypes, betas):
        if self.is_identity:
            return prototypes
        else:
            target = [tsf.apply_parameters(proto, beta) for tsf, proto, beta in zip(self.tsf_sequences, prototypes, betas)]
            return stack(target, dim=1)

    def add_noise(self, noise_scale=0.001):
        for i in range(len(self.tsf_sequences)):
            self.tsf_sequences[i].load_with_noise(self.tsf_sequences[i], noise_scale=noise_scale)

    def step(self):
        if not self.is_identity:
            [tsf_seq.step() for tsf_seq in self.tsf_sequences]

    def activate_all(self):
        if not self.is_identity:
            [tsf_seq.activate_all() for tsf_seq in self.tsf_sequences]

    @property
    def only_id_activated(self):
        return self.tsf_sequences[0].only_id_activated

    def set_optimizer(self, opt):
        self.optimizer = opt

    def __len__(self):
        return len(self.tsf_sequences)


class TransformationChain(nn.Module):
    def __init__(self, general_cfg, transformation_cfg):
        super().__init__()
        self.tsf_names = transformation_cfg['ops']
        self.n_tsf = len(self.tsf_names)
        tsfs = [self.get_module(op)(**general_cfg, **transformation_cfg.get(op, {})) for op in self.tsf_names]
        self.tsf_modules = nn.ModuleList(tsfs)

        # Curriculum
        self.activations = list(general_cfg.get('curriculum_learning', self.n_tsf*[True]))
        self.cur_milestone = 1
        assert len(self.activations) == self.n_tsf and all(isinstance(a, bool) for a in self.activations)

    @staticmethod
    def get_module(name):
        return {
            # standard
            'identity': IdentityModule,
            'color': ColorModule,
            'position': PositionModule,
        }[name]

    @property
    def is_identity(self):
        return all(tsf == 'identity' for tsf in self.tsf_names)

    def predict_parameters(self, features):
        betas = []
        for module, activated in zip(self.tsf_modules, self.activations):
            if activated:
                predicted_beta = module.predict_parameters(features)
                betas.append(predicted_beta)
        
        return torch.cat(betas, dim=1)

    def set_optimizer(self, opt):
        self.optimizer = opt

    def apply_parameters(self, x, beta):
        dims = [d.dim_parameters for d, act in zip(self.tsf_modules, self.activations) if act]
        betas = torch.split(beta, dims, dim=1)
        for module, activated, beta in zip(self.tsf_modules, self.activations, betas):
            if activated:
                x = module.transform(x, beta)
        return x

    def load_with_noise(self, tsf_seq, noise_scale):
        for k in range(self.n_tsf):
            self.tsf_modules[k].load_with_noise(tsf_seq.tsf_modules[k], noise_scale)

    def activate(self, idx):
        self.activations[idx] = True

    def activate_all(self):
        for k in range(self.n_tsf):
            self.activate(k)
        self.next_act_idx = self.n_tsf

    @property
    def only_id_activated(self):
        for m, act in zip(self.tsf_modules, self.activations):
            if not isinstance(m, (IdentityModule,)) and act:
                return False
        return True
