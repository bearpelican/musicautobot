## MusicAutobot

Using Deep Learning to generate pop music! 

Please visit our really cool web app - [musicautobot.com](musicautobot.com)

### Overview

Recent advances in NLP have produced amazing [results](https://transformer.huggingface.co/) in generating text. 
[Transformer](http://jalammar.github.io/illustrated-transformer/) architecture is a big reason behind this.

This project aims to leverage these powerful language models and apply them to music. It's built on top of the fast.ai [library](https://github.com/fastai/fastai)

### Implementation

1. [MusicTransformer](musicautobot/music_transformer) - This basic model uses [Transformer-XL](https://github.com/kimiyoung/transformer-xl) to take a sequence of music notes and predict the next note.
2. [MultitaskTransformer](musicautobot/multitask_transformer) - Built on top of MusicTransformer, this model is trained on multiple tasks:
 * Next Note Prediction (same as MusicTransformer)
 * [BERT](https://github.com/google-research/bert) Token Masking
 * Sequence To Sequence Translation - Using chords to predict melody and vice versa.
```
These extra tasks allow us to generate some really cool predictions. 
1. Harmonization
2. Generate melodies from existing chord progression
3. New melody in the rhythm of another song
4. Same song with different rhythm
```
Example notebook [here](notebooks/multitask_transformer/Generate.ipynb).

### Example Notebooks

1. MusicTransformer
 * [Train](notebooks/music_transformer/Train.ipynb) - End to end example on how to create a dataset from midi files and train a model from scratch
 * [Generate](notebooks/music_tranformer/Generate.ipynb) - Loads a pretrained model and shows how to generate/predict new notes
 
2. MultitaskTransformer
 * [Train](notebooks/music_transformer/Train.ipynb) - End to end example on creating a seq2seq and masked dataset for multitask training.
 * [Generate](notebooks/music_tranformer/Generate.ipynb) - Loads a pretrained model and shows how to harmonize, generate new melodies, and remix existing songs.
 
3. Data Encoding
 * [Midi2Tensor](notebooks/data_encoding/Midi2Tensor.ipynb) - Shows how the libary internally encodes midi files to tensors for training.
 * [MusicItem](notebooks/data_encoding/MusicItem-Transforms.ipynb) - MusicItem is a wrapper that makes it easy to manipulate midi data. Convert midi to tensor, apply data transformations, even play music or display the notes within browser.
 
### Library

* [musicautobot](musicautobot)
 * [numpy_encode](musicautobot/numpy_encode.py) - Leverages music21's incredible [library](https://web.mit.edu/music21/) to transform midi files into tensors for training
 * [config](musicautobot/config.py) - Default model parameters
 * [vocab](musicautobot/vocab.py) - Dictionary for tokenizing midi to tensor. 
 * [music_transformer](musicautobot/music_transformer) - File structure similar to fastai's library.
   * Learner, Model, Transform - MusicItem, Dataloader
 * [multitask_transformer](musicautobot/multitask_transformer) - File structure similar to fastai's library.
   * Learner, Model, Transform - MusicItem, Dataloader

### Scripts

These scripts are 
* [run_multitask.py](scripts/run_multitask.py) - CLI script for training multitask model
 * Usage: `python run_multitask.py --epochs 14 --save multitask_model --batch_size=16 --bptt=512 --lamb --data_parallel --lr 1e-4`
* [run_music_transformer.py](scripts/run_music_transformer.py) - CLI script for training music transformer.
 * Usage: `python run_multitask.py --epochs 14 --save music_model --batch_size=16 --bptt=512 --lr 1e-4`
* [run_ddp.sh](scripts/run_ddp.sh) - Helper method to train with multiple GPUs. Only works with run_music_transformer.py
 * Usage: `SCRIPT=run_multitask.py bash run_ddp.sh --epochs 14 --save music_model --batch_size=16 --bptt=512 --lr 1e-4`

### Installation

Insall anaconda: https://www.anaconda.com/distribution/  

`git clone https://github.com/bearpelican/musicautobot.git`

`cd musicautobot`

`conda env update -f environment.yml`

`source activate musicautobot`

#### Musescore
If you want to be able to show scores in a jupyter notebook, install musescore:  

MacOS:  
Download here: https://musescore.org/en/download  

Ubuntu:  
`sudo apt-get install musescore` 


## Data

Unfortunately I cannot provide the dataset used for training the model.

For some good data sources, please look here:

* [Classical Archives](https://www.classicalarchives.com/)
* [Reddit](https://www.reddit.com/r/datasets/comments/3akhxy/the_largest_midi_collection_on_the_internet/)
* [wayne391](https://github.com/wayne391/Lead-Sheet-Dataset)
* [Lakh](https://colinraffel.com/projects/lmd/)


## Credits

This project is built on top of [fast.ai's](https://github.com/fastai/fastai) powerful deep learning library - which is built on top of pytorch


