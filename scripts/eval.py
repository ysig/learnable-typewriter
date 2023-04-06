import os, sys
from os.path import dirname, abspath, join
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

sys.path.append(join(dirname(dirname(abspath(__file__)))))
from learnable_typewriter.utils.loading import load_pretrained_model
from learnable_typewriter.evaluate.quantitative.sprite_matching.metrics import error_rate 
from learnable_typewriter.data.dataloader import collate_fn_pad_to_max

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_figure(gt, rec, seg, text, output_dir, i, j):
    fig = make_subplots(3, 1)

    # We use go.Image because subplots require traces, whereas px functions return a figure
    fig.add_trace(go.Image(z=gt), 1, 1)
    fig.add_trace(go.Image(z=rec), 2, 1)
    fig.add_trace(go.Image(z=seg), 3, 1)
    fig.update_layout(autosize=False, height=max(gt.size[1],  rec.size[1], seg.size[1])*7, width=max(gt.size[0], rec.size[0], seg.size[0]))
    kargs = {'title_font': dict(size=10)}
    fig.update_layout(title={'text': text, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, **kargs)

    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(f"{output_dir}/{i}_{j}.png")

def plot_sprites(names, images, fp):
    max_row = int(np.ceil(np.sqrt(len(images))))
    blk = Image.new("RGB", images[0].size, (255, 255, 255))
    subplot_titles = names + (max_row**2 - len(images))*[""]
    fig = make_subplots(rows=max_row, cols=max_row, subplot_titles=tuple(subplot_titles))
    fig.update_yaxes(showticklabels=False, automargin=True) # hide all the xticks
    fig.update_xaxes(showticklabels=False, automargin=True) # hide all the xticks
    fig.update_layout(autosize=False, height=images[0].size[1]*max_row*2, width=images[0].size[0]*2*max_row)
    fig.update_annotations(font_size=8)

    for i in range(max_row):
        for j in range(max_row):
            linear_idx = i*max_row + j
            if linear_idx >= len(images):
                image = blk
            else:
                image = images[linear_idx]
            fig.add_trace(go.Image(z=image.convert('RGB')), i+1, j+1)
    
    fig.write_image(f"{fp}")

def run(args):
    trainer = load_pretrained_model(path=args.input_path, device=str(args.device), kargs=args.kargs)
    trainer.evaluate_only = True
    trainer.train_end = True

    if args.idx_sample is not None and len(args.idx_sample):
        topil = ToPILImage()
        dataloader = getattr(trainer, f'{args.split}_loader')

        assert hasattr(args, 'output_path')
        assert len(args.idx_dataloader) == len(args.idx_sample)
        transcribe = (trainer.transcribe if trainer.supervised else trainer.transcribe_unsupervised)
        for i, j in zip(map(int, args.idx_dataloader), map(int, args.idx_sample)):
            dataset = dataloader[i].dataset
            supervised = dataset.supervised
            x = collate_fn_pad_to_max([dataset[j]], supervised=supervised)
            obj = trainer.decompositor(x)
            gt = topil(x['x'].cpu()[0])
            rec = topil(obj['reconstruction'].cpu()[0])
            seg = topil(obj['segmentation'].cpu()[0])
            y = trainer.inference(x)[0]
            map_pd = transcribe
            map_gt = dict((trainer.transcribe if trainer.supervised else trainer.transcribe_dataset))
            map_gt[-1] =  '_'

            cer = error_rate([(y, x['y'][0])], average=False, delim=dataset.space, sep=dataset.sep, map_pd=map_pd, map_gt=map_gt)
            text = 'pred: ' + cer['texts'][0] + ' <br> gt: ' + cer['gt'][0] + ' <br> ' + str(round(cer['cer'][0], 4)*100) + '%'
            make_figure(gt, rec, seg, text, args.output_path, i, j)

    if args.plot_sprites:
        names, images = [], []
        masks = trainer.model.masks
        for i in transcribe.keys():
            names.append(transcribe[i])
            images.append(topil(masks[i]))
        
        idxs = np.argsort(names)
        names = [names[i] for i in idxs]
        images = [images[i] for i in idxs]
        plot_sprites([names[i] for i in idxs], [images[i] for i in idxs], f'{args.output_path}/sprites.png')

    if args.eval_reco:
        trainer.log_train_metrics()
        trainer.log_val_metrics()

    if args.eval:
        print('Evaluating Error Rate')
        trainer.error_rate(False)

    if args.eval_best:
        print('Evaluating Error Rate for the Best Model')
        trainer.error_rate(True)

if __name__ == "__main__":
    import argparse, torch

    parser = argparse.ArgumentParser(description='Generic tool for quantitative (reco-error, error-rate) and qualitative (reconstruction, segmentation, sprites) evaluation.')
    parser.add_argument('-i', "--input_path", required=True, default=None, help='Model Path')
    parser.add_argument('-d', "--device", default=(0 if torch.cuda.is_available() else 'cpu'), type=str, help='Cuda ID')
    parser.add_argument("--eval", action='store_true', help='Evaluate error rate in last model.')
    parser.add_argument("--eval_reco", action='store_true', help='Evaluate reconstruction error in all splits.')
    parser.add_argument("--eval_best", action='store_true', help='Evaluate error rate on the best model.')
    parser.add_argument('-o', "--output_path", required=False, default='figures', type=str, help='Save Output dir')
    parser.add_argument('-s', "--split", default='val', type=str, help='dataset split')
    parser.add_argument('-id', "--idx_dataloader", nargs="+", required=False, help='List of indexes of the dataloaders that will plot images from.')
    parser.add_argument('-is', "--idx_sample", nargs="+", required=False, help='List of indexes of the samples for each dataloader that will plot images from.')
    parser.add_argument("--plot_sprites", action='store_true', help='Plot a sprite-grid')
    parser.add_argument("--kargs", type=str, default='training.batch_size=16', help='Override the loaded kwargs with an injected OmegaConf profile.')
    args = parser.parse_args()

    run(args)
