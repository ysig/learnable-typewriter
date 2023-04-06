import json
import random
from os import listdir
from os.path import join, isfile
from functools import partial
from PIL import Image

import numpy as np
import cv2
from skimage import exposure
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop


### REFACTOR WITH DATACLASS ###
class LineDataset(Dataset):
    """ The Line Dataset expects a very simple structure.
    Each dataset should contain:
        - a path location where we can find:
            - a folder of images denoted as 'images'
            - a file denoted as annotation.json containing:
              - 'file-name.ext': {'split': 'train'/'test'[, 'label': 'I like cats.']}
              where file-name should be any name found under images.
        Omitting this file will assume there is no val set.
        Omitting all labels will assume there are none.
        Omitting one label will result in a warning message.
    """
    n_channels = 3

    def __init__(self, path, height, split, alias, crop_width=None, space=' ', N_min=0, W_max=float('inf'), dataset_size=None, transcribe=None, sep='', supervised=False, padding=None, padding_value=None, filter_by_name=None):
        '''
        Inputs:
            - N_min : minimal number of occurences in the dataset for the least frequent character of each selected instance
            - W_max : maximal width for each selected instance after transformation
            - dataset_size : desired number of elements in the dataset
        '''
        self.alias = alias
        self.supervised = supervised
        self.sep = sep
        self.N_min = N_min
        self.W_max = W_max
        self.length_available = dataset_size

        self.path = path
        self.split = split
        self.space = space
        self.height = height

        self.filter_by_name = filter_by_name

        self.transcribe = transcribe
        if transcribe is not None:
            self.alphabet = set(transcribe.values())
        else:
            self.alphabet = set()

        self.files = self.extract()

        if transcribe is None:
            self.transcribe = dict(enumerate(sorted(self.alphabet)))

        self.size = len(self.files)
        self.crop_width = crop_width

        self.matching = {char: num for num, char in self.transcribe.items()}
        self.padding_value = padding_value

        if isinstance(self.padding_value, tuple) and all(((not isinstance(p, int)) and p < 1) for p in self.padding_value):
            self.padding_value = tuple(int(p*255) for p in self.padding_value)
        elif self.padding_value < 1 and isinstance(self.padding_value, float):
            self.padding_value = int(self.padding_value*255)

        self.build_transform()
        assert not (self.supervised and not self.has_labels), "If dataset is used in supervised mode it should contain labels."

    def build_transform(self):
        transform = []
        if self.cropped:
            transform.append(RandomCrop((self.height, self.crop_width), pad_if_needed=True, fill=self.padding_value, padding_mode='constant'))
        self.transform = Compose(transform)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        x = Image.open(join(self.image_path, self.files[i])).convert('RGB')
        x = x.resize((int(self.height * x.size[0] / x.size[1]), self.height))
        x = self.transform(x)
        return x, self.get_label(i)

    @property
    def has_labels(self):
        return self.annotation is not None

    @property
    def annotation_path(self):
        return join(self.path, 'annotation.json')

    @property
    def image_path(self):
        return join(self.path, 'images')

    def process_transcription(self,raw_transcription):
        transcription = raw_transcription.replace(self.space, '')
        if self.sep != '':
            transcription = transcription.split(self.sep)
        return transcription

    ################### NEEDS REFACTORING ####################
    def filter_files_func_cnt(self,kv):
        '''Returns True if instance represented by kv is of desired width/number of occurences for least frequent character,
        and if dataset isn't full, else False'''
        N = kv[1].get('N', -1)
        label = self.process_transcription(kv[1]['label'])
            
        if self.length_available > 0 :
            if N >= self.N_min or N == -1:
                if self.W_max == float('inf') :
                    if self.transcribe is None:
                        for char in label:
                            self.alphabet.add(char)
                    self.length_available -= 1
                    return True
                else : 
                    x = Image.open(join(self.image_path, kv[0])).convert('RGB')
                    width, height = x.size[0], x.size[1]
                    new_width  = int(self.height * width / height)

                    if new_width <= self.W_max:
                        if self.transcribe is None :
                            for char in label:
                                self.alphabet.add(char)
                        self.length_available -= 1
                        return True
        return False
    
    def filter_files_func(self,kv):
        '''Returns True if instance represented by kv is of desired width/number of occurences for least frequent character,
        else False'''
        N = kv[1].get('N', -1)
        label = self.process_transcription(kv[1]['label'])

        if N >= self.N_min or N == -1:
            if self.W_max == float('inf'):
                if self.transcribe is None:
                    for char in label:
                        self.alphabet.add(char)
                return True
            else : 
                x = Image.open(join(self.image_path, kv[0])).convert('RGB')
                width, height = x.size[0], x.size[1]
                new_width  = int(self.height * width / height)

                if new_width <= self.W_max:
                    if self.transcribe is None:
                        for char in label:
                            self.alphabet.add(char)
                    return True
        return False
    ###############################################################

    def read_annotations_json(self):
        if not isfile(self.annotation_path):
            if self.supervised:
                raise RuntimeError('Annotation not found at path {}'.format(self.path))
            else:
                self.annotation = None
        else:
            with open(self.annotation_path) as f:
                self.annotation = json.load(f)

            data = dict(filter(lambda kv: kv[1]['split'] == self.split, self.annotation.items()))
            any_label = any('label' in v for _, v in data.items())
            dico = (dict(filter(lambda kv: 'label' in kv[1], data.items())) if any_label else data)
            filter_func = self.filter_files_func_cnt if self.length_available is not None else self.filter_files_func
            dico = dict(filter(filter_func, dico.items()))  #filters number of occurences of characters

            return dico

    def extract_split(self, keep):
        if keep is None:
            return listdir(self.image_path) if self.split == 'train' else []
        else:
            return [p for p in listdir(self.image_path) if p in set(keep)]

    @property
    def cropped(self):
        return self.crop_width is not None

    def get_label(self, i):
        if self.annotation is None or self.cropped:
            return -1
        else:
            transcription = self.process_transcription(self.annotation.get(self.files[i], -1)['label'])
            return [self.matching.get(char, -1) for char in transcription]

    def extract(self):
        image_list = self.read_annotations_json()
        files = sorted(self.extract_split(image_list))
        if self.filter_by_name is None:
            return files
        else:
            return [f for f in files if self.filter_by_name in f]


class ExemplarDataset(Dataset):
    """ Exemplar Dataset creates a random synset from a set of examples. """
    def __init__(self, path, height, split, alias, W_min=256, W_max=1024, space=' ', sep='', seed=42, num_samples=1000, supervised=True, transcribe=None, ext='jpg', use_matching=False):
        self.sep = sep
        assert ext in {'jpg', 'png'}
        self.ext = '.' + ext
        self.W_min, self.W_max = W_min, W_max

        self.path = path
        self.sep, self.space = sep, space
        self.H = height
        self.seed = 42
        self.size_place = self.H
        self.letter_distance_px = self.H//16
        self.spacing_px = self.H//8 + self.H//32
        self.split = split
        self.alias = alias
        self.num_samples = num_samples
        self.supervised = supervised
        self.transcribe = transcribe
        self.use_matching = use_matching
        self.build()
        #assert all(len(self.data[i][1]) for i in range(len(self.data)))

    def __len__(self):
        return self.num_samples

    @property
    def cropped(self):
        return False

    def build(self):
        self.make_vocabulary()

    def preitem(self, i):
        if self.split == 'val':
            # hack to have the same val dataset
            np.random.seed(i)
            random.seed(i)
        elif self.split == 'test':
            # hack to have the same val dataset
            np.random.seed(i+len(self))
            random.seed(i+len(self))

        return self.make_sequence()

    def __getitem__(self, i):
        image, label = self.preitem(i)
        label = ([-1] if self.cropped else label)
        return image, label

    def resize_h(self, img):
        img = img.convert('RGB')
        if img.size[1] == self.size_place:
            return img
        hpercent = (self.size_place/float(img.size[1]))
        w = int((float(img.size[0])*float(hpercent)))
        img = img.resize((w, self.size_place), Image.Resampling.LANCZOS)
        return img

    def make_vocabulary(self):
        if self.transcribe is None:
            self.vocabulary = {image.replace(self.ext, ''): self.resize_h(Image.open(join(self.path, image))) for image in listdir(self.path)}
            self.alphabet = list(sorted(self.vocabulary.keys()))
            self.matching = {c: i for i, c in enumerate(self.alphabet)} 
            self.transcribe = {i: c for c, i in self.matching.items()}
        else:
            self.matching = {i: c for c, i in self.transcribe.items()}
            self.alphabet = list(sorted(self.matching.keys()))
            self.vocabulary = {a: self.resize_h(Image.open(join(self.path, f'{a}{self.ext}'))) for a in self.alphabet}

    def make_sequence(self):
        data = [], []
        while len(data[1]) < 4:
            data = self.make_sequence_()
        return data

    def random_pm(self, center, offset):
        return random.randint(max(center - offset, 0), center + offset)

    def random_in(self, low, high, gauss=True):
        if gauss:
            # 3σ
            return int(low + (high-low)*(np.clip(random.gauss(0, 1), -3, 3) + 3)/6)
        else:
            return random.randint(0, min(high - low, 0)) + low
        
    def make_sequence_(self):
        image = np.zeros((self.H, self.W_max, 3)).astype(np.uint8)
        mask = (np.ones((self.H, self.W_max))*255).astype(np.uint8)
        length = self.W_min + (self.W_max-self.W_min)*(np.clip(random.gauss(0, 1), -3, 3) + 3)/6
        label, start_w = [], 0
        num = len(self.alphabet)

        if self.use_matching:
            reference = np.array(self.vocabulary[self.alphabet[np.random.choice(len(self.alphabet), size=1)[0]]])
            match = partial(exposure.match_histograms, reference=reference, multichannel=True)
        else:
            match = lambda x: x

        while True:
            idx = np.random.choice(range(num+1))
            if idx != num:
                l = self.alphabet[idx]
                img = self.vocabulary[l]
                
                w, h = img.size
                if start_w + w >= length:
                    break

                label.append(idx)
                random_small_h = self.random_in(0, self.H - h)
                start_h = random_small_h
                end_h = h + random_small_h
                end_w = w + start_w
        
                image[start_h:end_h, start_w:end_w] = match(np.array(img))
                mask[start_h:end_h, start_w:end_w] = 0
                start_w += w
            else:
                start_w += self.random_pm(self.spacing_px, self.spacing_px//8)

            start_w += self.random_pm(self.letter_distance_px, self.letter_distance_px//8)
            if start_w >= self.W_max:
                break

        max_w = min(start_w, self.W_max)
        image = cv2.inpaint(image[:, :max_w, :], mask[:, :max_w], 3, cv2.INPAINT_TELEA)
        return Image.fromarray(image), label


# TODO use for debugging drop later
if __name__ == "__main__":
    path = '/home/ysig/nicolas/learnable-typewriter-supervised/datasets/Google1000/exemplars'
    train = ExemplarDataset(path, height=64, split='train', alias='exemplars')
    val_1 = ExemplarDataset(path, height=64, split='val', alias='exemplars')
    val_2 = ExemplarDataset(path, height=64, split='val', alias='exemplars')
    
    a, b, c, d = [train.preitem(i) for i in range(10)], val_1.preitem(0), val_1.preitem(1), val_2.preitem(0)
    def make_figure(gt, rec, rec_2, seg, matching):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        texts = tuple(g[1] for g in gt)
        fig = make_subplots(len(gt) + 3, 1, subplot_titles=[''.join([matching[a] for a in t]) for t in texts + (seg[1], rec[1], rec_2[1])])
        fig.update_layout(autosize=False, height=gt[0][0].size[1]*2*(len(texts) + 2), width=1024)

        # We use go.Image because subplots require traces, whereas px functions return a figure
        for i in range(1, len(texts)+1):
            fig.add_trace(go.Image(z=gt[i-1][0]), i, 1)
        fig.add_trace(go.Image(z=seg[0]), len(texts)+1, 1)
        fig.add_trace(go.Image(z=rec[0]), len(texts)+2, 1)
        fig.add_trace(go.Image(z=rec_2[0]), len(texts)+3, 1)
        fig.write_image(f"debug-blueprint.png")

    make_figure(a, b, c, d, train.transcribe)
