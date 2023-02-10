# The Learnable Typewriter <br><sub>A Generative Approach to Text Line Analysis</sub>
Official PyTorch implementation of [The Learnable Typewriter: A Generative Approach to Text Line Analysis](https://imagine.enpc.fr/~siglidii/learnable-typewriter/).  
Authors: [Yannis Siglidis](https://imagine.enpc.fr/~siglidii/), [Nicolas Gonthier](https://perso.telecom-paristech.fr/gonthier/), [Julien Gaubil](https://juliengaubil.github.io/), [Tom Monnier](https://www.tmonnier.com/), [Mathieu Aubry](http://imagine.enpc.fr/~aubrym/).  
Research Institute: [Imagine](https://imagine.enpc.fr/), _LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France_

## Experimental Features :fire:
Experimental features are those that are not described in the paper, but which were rather interesting and may provide inspiration for future work.

### Unsupervised Regularizers
Close to publishing our work we have considered different types of regularizers to make unsupervised learning more meaningful.
We decided that the improvements we were getting from these regularizers were not in the end that essential to include them in the final paper.
However for potential future use and for practical purposes we describe them here theoretically and describe how they can be accessed from our code base.

#### Overlap
An undesirable behavior for image reconstruction is for the network to learn to reconstruct characters by placing one on top of the others.
To this end we propose a overlap penalization regularizer applied between different image layers:

$$\mathcal{L}_{\text{overlap}}= \lambda_{\textrm{over}} \sum_{t=1}^T \sum_{j=t+1}^T \mathbf{o}^\alpha_{t} \cdot \mathbf{o}^\alpha_{j}$$

We define it with the hyperparameter $\lambda_{\textrm{over}}$, accessed by setting `model.loss.overlap`, with a default value being 0.00001.

#### Frequency Loss
An undesirable behavior is for the network to use only a fraction of the sprites. We present this behavior by encouraging all sprites to be used at least at some frequency:

$$\mathcal{L}_{\text{freq}} =  1 - \frac{1}{\epsilon}\sum_{k=1}^{K+1}\min(\mu_{k}, \frac{\epsilon}{K+1})$$

where $\mu_{k} = \frac{1}{T} \sum\nolimits_{t=1}^{T} p_{k}(f_t)$ is the mean usage of sprite $k$, $p_{k}(f_t)$ is the probability of the sprite $s_k$ being associated to the feature $f_t$. 

We define it with the hyperparameter $\epsilon$, accessed by setting `model.loss.frequency`, with a default value being 0.1.

#### Sparsity Loss
A final undesirable behavior is for the network to learn to not use blanks as often as it could:

$$\mathcal{L}_{\text{sparse}} = \lambda_{\textrm{empty}} \mu_{K+1}$$

We define it with the hyperparameter $\lambda_{\textrm{empty}}$, accessed by setting `model.loss.sparse`, with a default value being 0.05.
Note, that for positive values of $\lambda_{\textrm{empty}}$ this loss only makes sense if its used together with the frequency loss.

In total you can find an example of configurations for all the aforementioned losses in:
```python
python scripts/train.py configs/unsupervised-copiale-reg.yaml
```
and then for the selected model:
```python
python scripts/fontenay.py -i <FONTENAY-MULTI-PATH> -o <OUTPUT-DIR> --filter FRAD021_15_H_183_000213 --invert_sprites
```
### Multiple Sprites Per Character
In the supplementary material we demonstrate a supervised experiment where we learn 2 characters per letter, by summing their respective probabilities when computing the ctc-loss.
It allow to capture more diverse handwriting, which is common case when dealing with different generations of writers.
```python
python scripts/train.py supervised-fontenay-multi.yaml
```

### `SequentialAdaptiveDataloader` 
Parsing lines of text is extremely suboptimal in terms of memory when performed in batches.  
When selecting a batch size, you must speculate of what is the memory upper limit when N random samples from the dataset will be batched.  

The worst memory footprint becomes `N*H*max([im.size()[-1] for im in images))`.  
If the lengths of the lines exhibit significant variance, then selecting a batch size based on the maximal line will leave the GPU, much more empty on average than intended.  
As a solution to this a `SequentialAdaptiveDataloader` has been designed that tries to fit as much elements to a fixed limit of `N*<factor>`.  

You can enable this feature by setting: `training.adaptive_dataloader=False`.
The optimization procedure (loss, hyperparameters) isn't designed to for area instead `N*W` instead of the batch-size `N`, so the performance isn't expected to be optimal.

> Receiving a cuda-error `device assert trigger`, is probably an alias for a `RuntimeError: CUDA Out of memory` error.

### Encoder-RNN
As we didn't want to encode language priors and to present an as simple architecture as possible, our proposed encoder doesn't contain recurrent elements.
However the traditional state-of-the-art for HTR e.g. [Puigcerver (2017)](http://jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf) uses an RNN.
For practical applications you can try using an RNN at the end of the encoder network by modifying the config file to:
```
model:
  encoder:
    rnn:
      type: 'gru' # or 'lstm'
      num_layers: 3
      dropout: 0
      bias: True
      bidirectional: True
```

### Multiple Dataset Training
You can train in multiple datasets, supervised and unsupervised.  
This can be helpful for partially labelled datasets.  
You can do it by setting:
```
  - dataset:
    - <dataset-1-tag>
    ...
    - <dataset-n-tag>
```

## TODOs :satellite:
This code base has taken multiple iterations and comes from different people from different places and times.
From its evolutionary nature, it can only imply spandrels, that it is elements that don't resist its general function but still are not optimized or fit to their best.
Here we list some of those:

- GaussianPool
GaussianPool was an idea for having covariant average pooling to reduce the dimensions of the ResNet output tensor.
It performs down-sampling as its kernel size is smaller than the stride size.
Although it worked better than AvgPooling it may not be optimal.

- Remove Layer Transform Notation
Layer transform is a terminology that comes from DTI-Sprites. It could be totally removed to a more understandable term and simplified as we only use constant colors.

- Simplifying the Transformation Module
We currently use a single transform per element. The old transformation code that the we import from DTI-Sprites is too complicated for our current use case.

- Overall memory optimization
Although the code has been cleaned as much as possible to be easy and joyful to read, a lot of information is transferred stored and logged.
Also when using large sprites banks selection could be performed in batches which would significantly lower the maximum memory usage.
