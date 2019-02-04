## Environment setup:  

#### Anaconda
Recommend installing anaconda: https://www.anaconda.com/distribution/  

After installation, create our midi environment:
`conda create -n midi python=3.7`
`conda activate midi`

#### Musescore
If you want to be able to show scores in a jupyter notebook, install musescore:  

MacOS:
Download here: https://musescore.org/en/download  
You may need to set music21 environment path in jupyter notebook:  
`music21.environment.set('musicxmlPath', '/Applications/MuseScore 3.app/Contents/MacOS/mscore')`
`music21.environment.set('musescoreDirectPNGPath', '/Applications/MuseScore 3.app/Contents/MacOS/mscore')`

Ubuntu:
`sudo apt-get install musescore`
`music21.environment.set('musicxmlPath', '/usr/bin/musescore')`
`music21.environment.set('musescoreDirectPNGPath', '/usr/bin/musescore')`


## Installation

`git clone git@github.com:bearpelican/pytorch_midi_generator.git`

`pip install -r requirements.txt`

`mkdir -p data/midi`

`cd data/midi`

`wget https://s3-us-west-2.amazonaws.com/ashaw-music/v3/midi_transcribe_v3_shortdur_wmodels.tar.gz`

`tar -xzvf midi_transcribe_v3_shortdur.tar.gz`

## Generating music

Start Jupyter notebook:  
`jupyter notebook`

Open up `ULMFit-2-testing.ipynb`

You should be able to run through all the cells


## Training

`SCRIPT=run_gpt.py bash run_multi.sh --path data/midi/midi_transcribe_v3_shortdur/ --batch_size 8 --lr .0001 --epochs 5 --save gpt/clc/v3ep50`


## Data

The `1-DataFormatting-#-XXXX.ipynb` notebooks are the steps to encode midi files to text to numerical tensors.

`data_collection` directory contains information on how midi data was scraped. In all, there are 33k midi files collected.

### Dataset
The dataset here `https://s3-us-west-2.amazonaws.com/ashaw-music/v3/midi_transcribe_v3_shortdur_wmodels.tar.gz` has already been transformed to text format.

Encoding format:
Take a looks at `Tutorial-EncodingFormat.ipynb` for better idea of how midi is encoded