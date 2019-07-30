## MusicAutobot

Using Deep Learning to generate pop music! 

Please visit our really cool web app - [musicautobot.com](musicautobot.com)

### Overview

Recent advances in NLP have produced amazing [results](https://transformer.huggingface.co/) in generating text. 
[Transformer](http://jalammar.github.io/illustrated-transformer/) architecture is a big reason behind this.

This project aims to leverage these powerful language models and apply them to music. It's built on top of the fast.ai [library](https://github.com/fastai/fastai)

### Implementation

1. [MusicTransformer](src/music_transformer) - This basic model uses [Transformer-XL](https://github.com/kimiyoung/transformer-xl) to take a sequence of music notes and predict the next note.
2. [MultitaskTransformer](src/multitask_transformer) - Built on top of MusicTransformer, this model is trained on multiple tasks:
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

#### Example Notebooks

1. MusicTransformer
 * [Train](notebooks/music_transformer/Train.ipynb) - End to end example on how to create a dataset from midi files and train a model from scratch
 * [Generate](notebooks/music_tranformer/Generate.ipynb) - Loads a pretrained model and shows how to generate/predict new notes
 
2. MultitaskTransformer
 * [Train](notebooks/music_transformer/Train.ipynb) - End to end example on creating a seq2seq and masked dataset for multitask training.
 * [Generate](notebooks/music_tranformer/Generate.ipynb) - Loads a pretrained model and shows how to harmonize, generate new melodies, and remix existing songs.
 
3. Data Encoding
 * [Midi2Tensor](notebooks/data_encoding/Midi2Tensor.ipynb) - Shows how the libary internally encodes midi files to tensors for training.
 * [MusicItem](notebooks/data_encoding/MusicItem-Transforms.ipynb) - MusicItem is a wrapper that makes it easy to manipulate midi data. Convert midi to tensor, apply data transformations, even play music or display the notes within browser.
 
#### Library

* [src](src) Rename to musicautobot
 * [numpy_encode](src/numpy_encode.py) - Leverages music21's incredible [library](https://web.mit.edu/music21/) to transform midi files into tensors for training
 * [config](src/config.py) - Default model parameters
 * [vocab](src/vocab.py) - Dictionary for tokenizing midi to tensor. 
 * [music_transformer](src/music_transformer) - File structure similar to fastai's library.
   * Learner, Model, Transform - MusicItem, Dataloader
 * [multitask_transformer](src/multitasl_transformer) - File structure similar to fastai's library.
   * Learner, Model, Transform - MusicItem, Dataloader

#### Scripts



#### Installation

#### Anaconda
Recommend installing anaconda: https://www.anaconda.com/distribution/  



`git clone git@github.com:bearpelican/pytorch_midi_generator.git`

`cd midi_generator`

`conda env update -f environment.yml`

`source activate midi`

#### Musescore
If you want to be able to show scores in a jupyter notebook, install musescore:  

MacOS:  
Download here: https://musescore.org/en/download  
You may need to set music21 environment path in jupyter notebook:  
```python
music21.environment.set('musicxmlPath', '/Applications/MuseScore 3.app/Contents/MacOS/mscore')
music21.environment.set('musescoreDirectPNGPath', '/Applications/MuseScore 3.app/Contents/MacOS/mscore')
```

Ubuntu:  
`sudo apt-get install musescore`  
```python
music21.environment.set('musicxmlPath', '/usr/bin/musescore')
music21.environment.set('musescoreDirectPNGPath', '/usr/bin/musescore')

os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QT_QPA_FONTDIR']='/usr/share/fonts'
```


This project is built on top of [fast.ai's](https://github.com/fastai/fastai) powerful deep learning library - which is built on top of pytorch


## Generating music

Start Jupyter notebook:  
`jupyter notebook`

Open up `notebook/examples/MultitaskGenerate.ipynb`

You should be able to run through all the cells


## Training

``

`SCRIPT=run_gpt.py bash run_multi.sh --path data/midi/midi_transcribe_v3_shortdur/ --batch_size 8 --lr .0001 --epochs 5 --save gpt/clc/v3ep50`


## Data

The `1-DataFormatting-#-XXXX.ipynb` notebooks are the steps to encode midi files to text to numerical tensors.

`data_collection` directory contains information on how midi data was scraped. In all, there are 33k midi files collected.

### Dataset
The dataset here `https://s3-us-west-2.amazonaws.com/ashaw-music/v3/midi_transcribe_v3_shortdur_wmodels.tar.gz` has already been transformed to text format.

Encoding format:
Take a looks at `Tutorial-EncodingFormat.ipynb` for better idea of how midi is encoded
