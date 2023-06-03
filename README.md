# Visual Search

Visual search is an advanced form of search that uses an image as a query instead of text.  
A Jupyter Notebook describing the training process and evaluation of a few models is included. In particular, the notebook shows the usage of pre-trained ResNet50 and VGG-16 models, which heads have been replaced with either an embedding followed by a classifier, or with an embedding only. The former kind of models can be trained using cross-entropy loss, while the latter can be trained using contrastive or triplet loss, along with a compatible miner; a function that creates image pairs or triplets to be used in the calculation of the loss.  
In particular, triplet loss operates using an anchor sample, a positive sample, and a negative sample. The anchor is the sample for which we want to adjust its embedding representation; as such the positive sample belongs to the same class, while the negative belongs to a different class. Therefore, the triplet loss aims to minimize the distances between anchors and positive samples, and maximize the distances between anchors and negative samples. 
Contrastive loss functions similarly, but only using either a positive or a negative sample, along with the anchor.  
A subset of the Oxford flowers 102 dataset, containing 24 species of flowers, was used throughout.

<img width="947" alt="Captura de Pantalla 2023-05-16 a las 0 51 35" src="https://github.com/RGonzLin/CV-Visual-Search/assets/65770155/1d5bc04b-0dc2-4432-8165-03e105031774">

## Key features
* Obtantion of embeddings for use in visual search
* Query an image to obtain similar ones
* Query a couple of images to evaluate if they belong to the same class
* Evaluate search performance using the mAP@k metric
